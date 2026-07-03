'' =====================================================================
''  Atlas - a character-level GPT language model, from scratch.
''  Pure FreeBASIC. No external ML libraries. Builds on Linux 64-bit and
''  Windows 32-bit from the same source.
''
''  Architecture:  token+positional embeddings
''                 -> NLAYER x ( causal multi-head self-attention
''                               + position-wise MLP ),
''                    each sublayer with pre-LayerNorm and a residual
''                 -> final LayerNorm -> linear head -> softmax
''
''  Forward pass, backward pass (all gradients hand-derived), the AdamW
''  optimizer, the tokenizer and the chat loop are all here.
''
''  Training is multi-threaded: the mini-batch is split across NTHREAD
''  worker threads, each with its own activation workspace and its own
''  gradient buffers; the gradients are reduced after every step.
''  Compile with -mt (see build scripts).
''
''  Usage:
''     atlas train [steps]   - train from data/corpus.txt, save model.bin
''     atlas chat            - load model.bin and chat
''     atlas                 - chat if model.bin exists, else train
'' =====================================================================

#include "crt.bi"   '' memset, fgets, fflush, stdin, stdout

'' isatty() is not exposed by crt.bi; declare it (libc on Linux, MinGW on Windows).
Extern "C"
    Declare Function isatty (ByVal fd As Long) As Long
End Extern

'' ---------------------------------------------------------------------
''  Model hyper-parameters (compile-time constants).  Edit + rebuild to
''  scale the model.  HEAD must divide NEMB; BATCH should be >= NTHREAD.
'' ---------------------------------------------------------------------
Const As Integer BLOCK  = 128         '' context length (tokens of history)
Const As Integer NEMB   = 192         '' embedding / residual-stream width
Const As Integer NHEAD  = 8           '' attention heads
Const As Integer HEAD   = NEMB \ NHEAD '' width per head
Const As Integer NLAYER = 6           '' transformer blocks
Const As Integer FFN    = 4 * NEMB    '' MLP hidden width

Const As Integer NTHREAD = 8          '' worker threads (use ~= CPU cores)

Const As Single  LN_EPS   = 1e-5
Const As Single  INIT_STD = 0.02

'' Training hyper-parameters
Const As Integer BATCH      = 16      '' sequences per optimizer step
Const As Single  LR_MAX     = 8e-4    '' peak learning rate
Const As Single  LR_MIN     = 8e-5    '' cosine floor
Const As Integer WARMUP     = 200
Const As Single  WD         = 0.01    '' weight decay (matrices only)
Const As Single  GRAD_CLIP  = 1.0
Const As Single  ADAM_B1    = 0.9
Const As Single  ADAM_B2    = 0.999
Const As Single  ADAM_EPS   = 1e-8

Const As String  MODEL_FILE  = "model.bin"
Const As String  CORPUS_FILE = "data/corpus.txt"

'' ---------------------------------------------------------------------
''  Parameter registry.  Every learnable tensor is registered here so the
''  optimizer, initializer and save/load iterate uniformly.  Each param
''  also carries one gradient buffer PER worker thread; they are summed
''  into .grad after each step.
'' ---------------------------------------------------------------------
Type ParamT
    data   As Single Ptr
    grad   As Single Ptr               '' reduced (summed) gradient
    gradT(NTHREAD-1) As Single Ptr      '' per-thread gradient
    m      As Single Ptr               '' Adam 1st moment
    v      As Single Ptr               '' Adam 2nd moment
    n      As Integer
    wd     As Integer                  '' apply weight decay? (1/0)
    nm     As String
End Type

Const As Integer MAXPARAM = 512
Dim Shared As ParamT params(MAXPARAM-1)
Dim Shared As Integer nparams = 0

Function newParam(nm As String, n As Integer, applyWd As Integer) As Integer
    Dim As Integer i = nparams
    With params(i)
        .n    = n
        .wd   = applyWd
        .nm   = nm
        .data = Callocate(n, SizeOf(Single))
        .grad = Callocate(n, SizeOf(Single))
        .m    = Callocate(n, SizeOf(Single))
        .v    = Callocate(n, SizeOf(Single))
        For tid As Integer = 0 To NTHREAD-1
            .gradT(tid) = Callocate(n, SizeOf(Single))
        Next
    End With
    nparams += 1
    Return i
End Function

'' Global parameter indices ------------------------------------------------
Dim Shared As Integer P_tok, P_pos, P_lnfg, P_lnfb, P_headw, P_headb
Dim Shared As Integer P_ln1g(NLAYER-1), P_ln1b(NLAYER-1)
Dim Shared As Integer P_qkvw(NLAYER-1), P_qkvb(NLAYER-1)
Dim Shared As Integer P_projw(NLAYER-1), P_projb(NLAYER-1)
Dim Shared As Integer P_ln2g(NLAYER-1), P_ln2b(NLAYER-1)
Dim Shared As Integer P_fcw(NLAYER-1),  P_fcb(NLAYER-1)
Dim Shared As Integer P_mpw(NLAYER-1),  P_mpb(NLAYER-1)

'' Vocabulary --------------------------------------------------------------
Dim Shared As Integer vocabSize = 0
Dim Shared As Integer id2ch(255)     '' id  -> byte value
Dim Shared As Integer ch2id(255)     '' byte -> id (-1 if unused)

Dim Shared As Single  scaleAtt       '' 1/sqrt(HEAD)

'' ---------------------------------------------------------------------
''  Per-thread workspace: all activations for ONE sequence, plus the
''  activation-gradient scratch used by the backward pass.  One of these
''  exists per worker thread so threads never touch each other's memory.
'' ---------------------------------------------------------------------
Type Workspace
    '' forward activations
    g_x(NLAYER, BLOCK-1, NEMB-1)       As Single   '' residual stream per block
    g_ln1(NLAYER-1, BLOCK-1, NEMB-1)   As Single
    g_ln1mu(NLAYER-1, BLOCK-1)         As Single
    g_ln1rs(NLAYER-1, BLOCK-1)         As Single
    g_q(NLAYER-1, NHEAD-1, BLOCK-1, HEAD-1) As Single
    g_k(NLAYER-1, NHEAD-1, BLOCK-1, HEAD-1) As Single
    g_v(NLAYER-1, NHEAD-1, BLOCK-1, HEAD-1) As Single
    g_att(NLAYER-1, NHEAD-1, BLOCK-1, BLOCK-1) As Single
    g_attout(NLAYER-1, BLOCK-1, NEMB-1) As Single
    g_amid(NLAYER-1, BLOCK-1, NEMB-1)  As Single
    g_ln2(NLAYER-1, BLOCK-1, NEMB-1)   As Single
    g_ln2mu(NLAYER-1, BLOCK-1)         As Single
    g_ln2rs(NLAYER-1, BLOCK-1)         As Single
    g_h(NLAYER-1, BLOCK-1, FFN-1)      As Single
    g_hact(NLAYER-1, BLOCK-1, FFN-1)   As Single
    g_final(BLOCK-1, NEMB-1)           As Single
    g_lnfmu(BLOCK-1)                   As Single
    g_lnfrs(BLOCK-1)                   As Single
    '' activation gradients
    dstream(BLOCK-1, NEMB-1)           As Single
    d_final(BLOCK-1, NEMB-1)           As Single
    d_attout(BLOCK-1, NEMB-1)          As Single
    d_qkv(BLOCK-1, 3*NEMB-1)           As Single
    tmpC(BLOCK-1, NEMB-1)              As Single
    tmpF(BLOCK-1, FFN-1)               As Single
    '' logits / probabilities scratch (<=256 vocab)
    wlog(255)                          As Single
    wp(255)                            As Double
End Type

Dim Shared As Workspace Ptr ws(NTHREAD-1)

Sub ensureWorkspaces(cnt As Integer)
    For i As Integer = 0 To cnt-1
        If ws(i) = 0 Then ws(i) = New Workspace
    Next
End Sub

'' =====================================================================
''  Small helpers
'' =====================================================================
Sub zeroPtr(p As Single Ptr, n As Integer)
    memset(p, 0, n * SizeOf(Single))
End Sub

Function randn() As Double
    Static As Integer haveSpare = 0
    Static As Double  spare
    If haveSpare Then
        haveSpare = 0
        Return spare
    End If
    Dim As Double u = Rnd, v = Rnd
    If u < 1e-12 Then u = 1e-12
    Dim As Double r  = Sqr(-2.0 * Log(u))
    Dim As Double th = 6.283185307179586 * v
    spare = r * Sin(th)
    haveSpare = 1
    Return r * Cos(th)
End Function

'' LayerNorm forward for one row of length NEMB.
Sub lnForward(x As Single Ptr, gain As Single Ptr, bias As Single Ptr, _
              outp As Single Ptr, ByRef mu As Single, ByRef rs As Single)
    Dim As Double m = 0
    For c As Integer = 0 To NEMB-1 : m += x[c] : Next
    m /= NEMB
    Dim As Double vv = 0
    For c As Integer = 0 To NEMB-1
        Dim As Double d = x[c] - m
        vv += d * d
    Next
    vv /= NEMB
    rs = 1.0 / Sqr(vv + LN_EPS)
    mu = m
    For c As Integer = 0 To NEMB-1
        outp[c] = (x[c] - m) * rs * gain[c] + bias[c]
    Next
End Sub

'' LayerNorm backward for one row.  dx is ACCUMULATED into the target.
Sub lnBackward(x As Single Ptr, dout As Single Ptr, mu As Single, rs As Single, _
               gain As Single Ptr, dgain As Single Ptr, dbias As Single Ptr, _
               dx As Single Ptr)
    Dim As Single xhat(NEMB-1), dxh(NEMB-1)
    Dim As Double dmean = 0, dvar = 0
    For c As Integer = 0 To NEMB-1
        xhat(c) = (x[c] - mu) * rs
        dxh(c)  = dout[c] * gain[c]
        dgain[c] += dout[c] * xhat(c)
        dbias[c] += dout[c]
        dmean += dxh(c)
        dvar  += dxh(c) * xhat(c)
    Next
    dmean /= NEMB
    dvar  /= NEMB
    For c As Integer = 0 To NEMB-1
        dx[c] += rs * (dxh(c) - dmean - xhat(c) * dvar)
    Next
End Sub

'' =====================================================================
''  Model construction / initialization
'' =====================================================================
Sub buildModel()
    scaleAtt = 1.0 / Sqr(HEAD)
    Dim As Single projStd = INIT_STD / Sqr(2.0 * NLAYER)

    P_tok = newParam("tok_emb", vocabSize * NEMB, 0)
    P_pos = newParam("pos_emb", BLOCK * NEMB, 0)
    For L As Integer = 0 To NLAYER-1
        P_ln1g(L)  = newParam("ln1g", NEMB, 0)
        P_ln1b(L)  = newParam("ln1b", NEMB, 0)
        P_qkvw(L)  = newParam("qkvw", (3*NEMB) * NEMB, 1)
        P_qkvb(L)  = newParam("qkvb", 3*NEMB, 0)
        P_projw(L) = newParam("projw", NEMB * NEMB, 1)
        P_projb(L) = newParam("projb", NEMB, 0)
        P_ln2g(L)  = newParam("ln2g", NEMB, 0)
        P_ln2b(L)  = newParam("ln2b", NEMB, 0)
        P_fcw(L)   = newParam("fcw", FFN * NEMB, 1)
        P_fcb(L)   = newParam("fcb", FFN, 0)
        P_mpw(L)   = newParam("mpw", NEMB * FFN, 1)
        P_mpb(L)   = newParam("mpb", NEMB, 0)
    Next
    P_lnfg  = newParam("lnfg", NEMB, 0)
    P_lnfb  = newParam("lnfb", NEMB, 0)
    P_headw = newParam("headw", vocabSize * NEMB, 1)
    P_headb = newParam("headb", vocabSize, 0)

    For i As Integer = 0 To nparams-1
        Dim As Single Ptr d = params(i).data
        Dim As Integer n = params(i).n
        Select Case params(i).nm
        Case "ln1g", "ln2g", "lnfg"
            For k As Integer = 0 To n-1 : d[k] = 1.0 : Next
        Case "ln1b", "ln2b", "lnfb", "qkvb", "projb", "fcb", "mpb", "headb"
            '' biases already zero
        Case "projw", "mpw"
            For k As Integer = 0 To n-1 : d[k] = randn() * projStd : Next
        Case Else
            For k As Integer = 0 To n-1 : d[k] = randn() * INIT_STD : Next
        End Select
    Next
End Sub

'' =====================================================================
''  FORWARD PASS  (fills workspace w for one sequence of length seqLen)
'' =====================================================================
Sub forward(ByRef w As Workspace, tok() As Integer, seqLen As Integer)
    Dim As Single Ptr tokW = params(P_tok).data
    Dim As Single Ptr posW = params(P_pos).data

    For t As Integer = 0 To seqLen-1
        Dim As Single Ptr te = @tokW[tok(t) * NEMB]
        Dim As Single Ptr pe = @posW[t * NEMB]
        For c As Integer = 0 To NEMB-1
            w.g_x(0, t, c) = te[c] + pe[c]
        Next
    Next

    For L As Integer = 0 To NLAYER-1
        Dim As Single Ptr ln1g = params(P_ln1g(L)).data
        Dim As Single Ptr ln1b = params(P_ln1b(L)).data
        Dim As Single Ptr qkvw = params(P_qkvw(L)).data
        Dim As Single Ptr qkvb = params(P_qkvb(L)).data
        Dim As Single Ptr prw  = params(P_projw(L)).data
        Dim As Single Ptr prb  = params(P_projb(L)).data
        Dim As Single Ptr ln2g = params(P_ln2g(L)).data
        Dim As Single Ptr ln2b = params(P_ln2b(L)).data
        Dim As Single Ptr fcw  = params(P_fcw(L)).data
        Dim As Single Ptr fcb  = params(P_fcb(L)).data
        Dim As Single Ptr mpw  = params(P_mpw(L)).data
        Dim As Single Ptr mpb  = params(P_mpb(L)).data

        '' pre-attention LayerNorm -> q,k,v projection
        For t As Integer = 0 To seqLen-1
            lnForward(@w.g_x(L, t, 0), ln1g, ln1b, @w.g_ln1(L, t, 0), _
                      w.g_ln1mu(L, t), w.g_ln1rs(L, t))
            Dim As Single Ptr ln = @w.g_ln1(L, t, 0)
            For j As Integer = 0 To 3*NEMB-1
                Dim As Single Ptr wr = @qkvw[j * NEMB]
                Dim As Single acc = qkvb[j]
                For c As Integer = 0 To NEMB-1 : acc += wr[c] * ln[c] : Next
                Dim As Integer part = j \ NEMB
                Dim As Integer off  = j Mod NEMB
                Dim As Integer hh   = off \ HEAD
                Dim As Integer dd   = off Mod HEAD
                Select Case part
                Case 0 : w.g_q(L, hh, t, dd) = acc
                Case 1 : w.g_k(L, hh, t, dd) = acc
                Case 2 : w.g_v(L, hh, t, dd) = acc
                End Select
            Next
        Next

        '' causal self-attention per head
        For hh As Integer = 0 To NHEAD-1
            For t As Integer = 0 To seqLen-1
                Dim As Single mx = -1e30
                For j As Integer = 0 To t
                    Dim As Single s = 0
                    For d As Integer = 0 To HEAD-1
                        s += w.g_q(L, hh, t, d) * w.g_k(L, hh, j, d)
                    Next
                    s *= scaleAtt
                    w.g_att(L, hh, t, j) = s
                    If s > mx Then mx = s
                Next
                Dim As Double sm = 0
                For j As Integer = 0 To t
                    Dim As Double e = Exp(w.g_att(L, hh, t, j) - mx)
                    w.g_att(L, hh, t, j) = e
                    sm += e
                Next
                Dim As Double inv = 1.0 / sm
                For j As Integer = 0 To t
                    w.g_att(L, hh, t, j) *= inv
                Next
                For d As Integer = 0 To HEAD-1
                    Dim As Single o = 0
                    For j As Integer = 0 To t
                        o += w.g_att(L, hh, t, j) * w.g_v(L, hh, j, d)
                    Next
                    w.g_attout(L, t, hh*HEAD + d) = o
                Next
            Next
        Next

        '' output projection + residual
        For t As Integer = 0 To seqLen-1
            Dim As Single Ptr ao = @w.g_attout(L, t, 0)
            For c As Integer = 0 To NEMB-1
                Dim As Single Ptr wr = @prw[c * NEMB]
                Dim As Single acc = prb[c]
                For c2 As Integer = 0 To NEMB-1 : acc += wr[c2] * ao[c2] : Next
                w.g_amid(L, t, c) = w.g_x(L, t, c) + acc
            Next
        Next

        '' pre-MLP LayerNorm -> MLP -> residual
        For t As Integer = 0 To seqLen-1
            lnForward(@w.g_amid(L, t, 0), ln2g, ln2b, @w.g_ln2(L, t, 0), _
                      w.g_ln2mu(L, t), w.g_ln2rs(L, t))
            Dim As Single Ptr ln = @w.g_ln2(L, t, 0)
            For k As Integer = 0 To FFN-1
                Dim As Single Ptr wr = @fcw[k * NEMB]
                Dim As Single acc = fcb[k]
                For c As Integer = 0 To NEMB-1 : acc += wr[c] * ln[c] : Next
                w.g_h(L, t, k) = acc
                w.g_hact(L, t, k) = acc / (1.0 + Exp(-1.702 * acc))
            Next
            Dim As Single Ptr ha = @w.g_hact(L, t, 0)
            For c As Integer = 0 To NEMB-1
                Dim As Single Ptr wr = @mpw[c * FFN]
                Dim As Single acc = mpb[c]
                For k As Integer = 0 To FFN-1 : acc += wr[k] * ha[k] : Next
                w.g_x(L+1, t, c) = w.g_amid(L, t, c) + acc
            Next
        Next
    Next

    '' final LayerNorm
    Dim As Single Ptr lnfg = params(P_lnfg).data
    Dim As Single Ptr lnfb = params(P_lnfb).data
    For t As Integer = 0 To seqLen-1
        lnForward(@w.g_x(NLAYER, t, 0), lnfg, lnfb, @w.g_final(t, 0), _
                  w.g_lnfmu(t), w.g_lnfrs(t))
    Next
End Sub

'' Logits for one position into w.wlog().
Sub logitsAt(ByRef w As Workspace, t As Integer)
    Dim As Single Ptr hw = params(P_headw).data
    Dim As Single Ptr hb = params(P_headb).data
    Dim As Single Ptr f  = @w.g_final(t, 0)
    For j As Integer = 0 To vocabSize-1
        Dim As Single Ptr wr = @hw[j * NEMB]
        Dim As Single acc = hb[j]
        For c As Integer = 0 To NEMB-1 : acc += wr[c] * f[c] : Next
        w.wlog(j) = acc
    Next
End Sub

'' =====================================================================
''  BACKWARD PASS.  Runs forward, computes mean cross-entropy loss over
''  the seqLen next-token targets, ACCUMULATES gradients into this
''  thread's params(*).gradT(tid).  Returns the loss (nats).
'' =====================================================================
Function backward(ByRef w As Workspace, tid As Integer, _
                  tok() As Integer, tgt() As Integer, seqLen As Integer) As Double
    forward(w, tok(), seqLen)

    Dim As Single Ptr hw  = params(P_headw).data
    Dim As Single Ptr dhw = params(P_headw).gradT(tid)
    Dim As Single Ptr dhb = params(P_headb).gradT(tid)

    zeroPtr(@w.d_final(0,0), BLOCK*NEMB)
    zeroPtr(@w.dstream(0,0), BLOCK*NEMB)

    Dim As Double loss = 0
    Dim As Single invT = 1.0 / seqLen

    '' softmax cross-entropy + head backward, per position
    For t As Integer = 0 To seqLen-1
        logitsAt(w, t)
        Dim As Single mx = -1e30
        For j As Integer = 0 To vocabSize-1
            If w.wlog(j) > mx Then mx = w.wlog(j)
        Next
        Dim As Double sm = 0
        For j As Integer = 0 To vocabSize-1
            w.wp(j) = Exp(w.wlog(j) - mx)
            sm += w.wp(j)
        Next
        Dim As Double inv = 1.0 / sm
        For j As Integer = 0 To vocabSize-1 : w.wp(j) *= inv : Next

        loss += -Log(w.wp(tgt(t)) + 1e-12)

        Dim As Single Ptr f = @w.g_final(t, 0)
        For j As Integer = 0 To vocabSize-1
            Dim As Single dl = (w.wp(j) - IIf(j = tgt(t), 1.0, 0.0)) * invT
            Dim As Single Ptr wr  = @hw[j * NEMB]
            Dim As Single Ptr dwr = @dhw[j * NEMB]
            For c As Integer = 0 To NEMB-1
                w.d_final(t, c) += dl * wr[c]
                dwr[c]          += dl * f[c]
            Next
            dhb[j] += dl
        Next
    Next

    '' final LayerNorm backward -> dstream
    Dim As Single Ptr lnfg  = params(P_lnfg).data
    Dim As Single Ptr dlnfg = params(P_lnfg).gradT(tid)
    Dim As Single Ptr dlnfb = params(P_lnfb).gradT(tid)
    For t As Integer = 0 To seqLen-1
        lnBackward(@w.g_x(NLAYER, t, 0), @w.d_final(t, 0), w.g_lnfmu(t), w.g_lnfrs(t), _
                   lnfg, dlnfg, dlnfb, @w.dstream(t, 0))
    Next

    '' transformer blocks in reverse
    For L As Integer = NLAYER-1 To 0 Step -1
        Dim As Single Ptr qkvw = params(P_qkvw(L)).data
        Dim As Single Ptr prw  = params(P_projw(L)).data
        Dim As Single Ptr fcw  = params(P_fcw(L)).data
        Dim As Single Ptr mpw  = params(P_mpw(L)).data
        Dim As Single Ptr ln1g = params(P_ln1g(L)).data
        Dim As Single Ptr ln2g = params(P_ln2g(L)).data

        Dim As Single Ptr dqkvw = params(P_qkvw(L)).gradT(tid)
        Dim As Single Ptr dqkvb = params(P_qkvb(L)).gradT(tid)
        Dim As Single Ptr dprw  = params(P_projw(L)).gradT(tid)
        Dim As Single Ptr dprb  = params(P_projb(L)).gradT(tid)
        Dim As Single Ptr dfcw  = params(P_fcw(L)).gradT(tid)
        Dim As Single Ptr dfcb  = params(P_fcb(L)).gradT(tid)
        Dim As Single Ptr dmpw  = params(P_mpw(L)).gradT(tid)
        Dim As Single Ptr dmpb  = params(P_mpb(L)).gradT(tid)
        Dim As Single Ptr dln1g = params(P_ln1g(L)).gradT(tid)
        Dim As Single Ptr dln1b = params(P_ln1b(L)).gradT(tid)
        Dim As Single Ptr dln2g = params(P_ln2g(L)).gradT(tid)
        Dim As Single Ptr dln2b = params(P_ln2b(L)).gradT(tid)

        '' === MLP branch ===  y = amid + mp(gelu(fc(ln2(amid))))
        zeroPtr(@w.tmpC(0,0), BLOCK*NEMB)
        For t As Integer = 0 To seqLen-1
            Dim As Single Ptr ha = @w.g_hact(L, t, 0)
            For k As Integer = 0 To FFN-1 : w.tmpF(t, k) = 0 : Next
            For c As Integer = 0 To NEMB-1
                Dim As Single dmo = w.dstream(t, c)
                Dim As Single Ptr wr  = @mpw[c * FFN]
                Dim As Single Ptr dwr = @dmpw[c * FFN]
                For k As Integer = 0 To FFN-1
                    w.tmpF(t, k) += dmo * wr[k]
                    dwr[k]       += dmo * ha[k]
                Next
                dmpb[c] += dmo
            Next
            For k As Integer = 0 To FFN-1
                Dim As Single x = w.g_h(L, t, k)
                Dim As Single sig = 1.0 / (1.0 + Exp(-1.702 * x))
                Dim As Single gp = sig + x * 1.702 * sig * (1.0 - sig)
                w.tmpF(t, k) *= gp
            Next
            Dim As Single Ptr ln = @w.g_ln2(L, t, 0)
            For k As Integer = 0 To FFN-1
                Dim As Single dh = w.tmpF(t, k)
                Dim As Single Ptr wr  = @fcw[k * NEMB]
                Dim As Single Ptr dwr = @dfcw[k * NEMB]
                For c As Integer = 0 To NEMB-1
                    w.tmpC(t, c) += dh * wr[c]
                    dwr[c]       += dh * ln[c]
                Next
                dfcb[k] += dh
            Next
        Next
        For t As Integer = 0 To seqLen-1
            lnBackward(@w.g_amid(L, t, 0), @w.tmpC(t, 0), w.g_ln2mu(L, t), w.g_ln2rs(L, t), _
                       ln2g, dln2g, dln2b, @w.dstream(t, 0))
        Next

        '' === Attention branch ===  amid = x + proj(attn(ln1(x)))
        zeroPtr(@w.d_attout(0,0), BLOCK*NEMB)
        For t As Integer = 0 To seqLen-1
            Dim As Single Ptr ao = @w.g_attout(L, t, 0)
            For c As Integer = 0 To NEMB-1
                Dim As Single da = w.dstream(t, c)
                Dim As Single Ptr wr  = @prw[c * NEMB]
                Dim As Single Ptr dwr = @dprw[c * NEMB]
                For c2 As Integer = 0 To NEMB-1
                    w.d_attout(t, c2) += da * wr[c2]
                    dwr[c2]           += da * ao[c2]
                Next
                dprb[c] += da
            Next
        Next

        zeroPtr(@w.d_qkv(0,0), BLOCK*3*NEMB)
        For hh As Integer = 0 To NHEAD-1
            For t As Integer = 0 To seqLen-1
                Dim As Single datt(BLOCK-1)
                Dim As Double dot = 0
                For j As Integer = 0 To t
                    Dim As Single s = 0
                    For d As Integer = 0 To HEAD-1
                        s += w.d_attout(t, hh*HEAD + d) * w.g_v(L, hh, j, d)
                    Next
                    datt(j) = s
                    dot += w.g_att(L, hh, t, j) * s
                Next
                For j As Integer = 0 To t
                    Dim As Single a  = w.g_att(L, hh, t, j)
                    Dim As Single ds = a * (datt(j) - dot) * scaleAtt
                    For d As Integer = 0 To HEAD-1
                        w.d_qkv(t, 0*NEMB + hh*HEAD + d) += ds * w.g_k(L, hh, j, d)
                        w.d_qkv(j, 1*NEMB + hh*HEAD + d) += ds * w.g_q(L, hh, t, d)
                        w.d_qkv(j, 2*NEMB + hh*HEAD + d) += a  * w.d_attout(t, hh*HEAD + d)
                    Next
                Next
            Next
        Next

        '' qkv linear backward
        zeroPtr(@w.tmpC(0,0), BLOCK*NEMB)
        For t As Integer = 0 To seqLen-1
            Dim As Single Ptr ln = @w.g_ln1(L, t, 0)
            For j As Integer = 0 To 3*NEMB-1
                Dim As Single dq = w.d_qkv(t, j)
                Dim As Single Ptr wr  = @qkvw[j * NEMB]
                Dim As Single Ptr dwr = @dqkvw[j * NEMB]
                For c As Integer = 0 To NEMB-1
                    w.tmpC(t, c) += dq * wr[c]
                    dwr[c]       += dq * ln[c]
                Next
                dqkvb[j] += dq
            Next
        Next
        For t As Integer = 0 To seqLen-1
            lnBackward(@w.g_x(L, t, 0), @w.tmpC(t, 0), w.g_ln1mu(L, t), w.g_ln1rs(L, t), _
                       ln1g, dln1g, dln1b, @w.dstream(t, 0))
        Next
    Next

    '' embedding backward
    Dim As Single Ptr dtok = params(P_tok).gradT(tid)
    Dim As Single Ptr dpos = params(P_pos).gradT(tid)
    For t As Integer = 0 To seqLen-1
        Dim As Single Ptr dte = @dtok[tok(t) * NEMB]
        Dim As Single Ptr dpe = @dpos[t * NEMB]
        For c As Integer = 0 To NEMB-1
            Dim As Single g = w.dstream(t, c)
            dte[c] += g
            dpe[c] += g
        Next
    Next

    Return loss * invT
End Function

'' =====================================================================
''  AdamW optimizer step (global grad-norm clipping), reads params.grad
'' =====================================================================
Sub adamStep(st As Integer, lr As Single)
    Dim As Double sq = 0
    For i As Integer = 0 To nparams-1
        Dim As Single Ptr g = params(i).grad
        For k As Integer = 0 To params(i).n-1 : sq += g[k] * g[k] : Next
    Next
    Dim As Single gnorm = Sqr(sq)
    Dim As Single clip = 1.0
    If gnorm > GRAD_CLIP Then clip = GRAD_CLIP / gnorm

    Dim As Single bc1 = 1.0 - (ADAM_B1 ^ st)
    Dim As Single bc2 = 1.0 - (ADAM_B2 ^ st)

    For i As Integer = 0 To nparams-1
        Dim As Single Ptr d = params(i).data
        Dim As Single Ptr g = params(i).grad
        Dim As Single Ptr m = params(i).m
        Dim As Single Ptr v = params(i).v
        Dim As Integer applyWd = params(i).wd
        For k As Integer = 0 To params(i).n-1
            Dim As Single gr = g[k] * clip
            m[k] = ADAM_B1 * m[k] + (1.0 - ADAM_B1) * gr
            v[k] = ADAM_B2 * v[k] + (1.0 - ADAM_B2) * gr * gr
            Dim As Single mh = m[k] / bc1
            Dim As Single vh = v[k] / bc2
            If applyWd Then d[k] -= lr * WD * d[k]
            d[k] -= lr * mh / (Sqr(vh) + ADAM_EPS)
        Next
    Next
End Sub

'' =====================================================================
''  Corpus loading + tokenization (tokens stored as bytes: 32-bit-safe
''  for very large corpora)
'' =====================================================================
Dim Shared As UByte dataArr()
Dim Shared As Integer nTok = 0

Sub loadCorpus()
    Dim As Integer f = FreeFile
    If Open(CORPUS_FILE For Binary Access Read As #f) <> 0 Then
        Print "ERROR: cannot open "; CORPUS_FILE : End 1
    End If
    Dim As LongInt sz = Lof(f)
    If sz = 0 Then Print "ERROR: corpus is empty" : End 1
    Dim As String s = Space(sz)
    Get #f, , s
    Close #f

    For i As Integer = 0 To 255 : ch2id(i) = -1 : Next
    vocabSize = 0
    For i As LongInt = 0 To sz-1
        Dim As Integer c = s[i]
        If ch2id(c) = -1 Then
            ch2id(c) = vocabSize
            id2ch(vocabSize) = c
            vocabSize += 1
        End If
    Next

    nTok = sz
    ReDim dataArr(nTok-1)
    For i As LongInt = 0 To sz-1
        dataArr(i) = ch2id(s[i])
    Next
    Print "corpus: "; nTok; " chars, vocab "; vocabSize; " symbols"
End Sub

'' =====================================================================
''  Save / load checkpoint
'' =====================================================================
Sub saveModel()
    Dim As Integer f = FreeFile
    If Open(MODEL_FILE For Binary Access Write As #f) <> 0 Then
        Print "ERROR: cannot write "; MODEL_FILE : Exit Sub
    End If
    '' Header uses Long (fixed 32-bit on every FB target) so the file is
    '' portable between platforms; FB's Integer is native word width
    '' (8 bytes on 64-bit, 4 on 32-bit) and would misalign cross-platform.
    Dim As Long magic = &h41544C34   '' "ATL4"
    Put #f, , magic
    Dim As Long cfg(5) = {BLOCK, NEMB, NHEAD, NLAYER, FFN, vocabSize}
    For i As Integer = 0 To 5 : Put #f, , cfg(i) : Next
    For i As Integer = 0 To vocabSize-1 : Put #f, , CLng(id2ch(i)) : Next
    '' bulk-write each tensor in one call (fast)
    For i As Integer = 0 To nparams-1
        Put #f, , params(i).data[0], params(i).n
    Next
    Close #f
    Print "saved -> "; MODEL_FILE
End Sub

Function loadModel() As Integer
    Dim As Integer f = FreeFile
    If Open(MODEL_FILE For Binary Access Read As #f) <> 0 Then Return 0
    Dim As Long magic : Get #f, , magic
    If magic <> &h41544C34 Then Close #f : Return 0   '' "ATL4"
    Dim As Long cfg(5)
    For i As Integer = 0 To 5 : Get #f, , cfg(i) : Next
    If cfg(0)<>BLOCK Or cfg(1)<>NEMB Or cfg(2)<>NHEAD Or cfg(3)<>NLAYER Or cfg(4)<>FFN Then
        Print "ERROR: model.bin was built with different dimensions."
        Close #f : Return 0
    End If
    vocabSize = cfg(5)
    For i As Integer = 0 To 255 : ch2id(i) = -1 : Next
    For i As Integer = 0 To vocabSize-1
        Dim As Long b : Get #f, , b
        id2ch(i) = b
        ch2id(id2ch(i)) = i
    Next
    buildModel()
    '' bulk-read each tensor in one call (fast)
    For i As Integer = 0 To nparams-1
        Get #f, , params(i).data[0], params(i).n
    Next
    Close #f
    Return 1
End Function

'' =====================================================================
''  Text generation (single-threaded, uses workspace 0)
'' =====================================================================
Function sampleNext(ctx() As Integer, ctxLen As Integer, temp As Single) As Integer
    Dim As Integer seqLen = ctxLen
    If seqLen > BLOCK Then seqLen = BLOCK
    Dim As Integer toks(BLOCK-1)
    Dim As Integer bpos = ctxLen - seqLen
    For i As Integer = 0 To seqLen-1 : toks(i) = ctx(bpos + i) : Next
    forward(*ws(0), toks(), seqLen)
    logitsAt(*ws(0), seqLen-1)
    Dim As Single mx = -1e30
    For j As Integer = 0 To vocabSize-1
        ws(0)->wlog(j) /= temp
        If ws(0)->wlog(j) > mx Then mx = ws(0)->wlog(j)
    Next
    Dim As Double sm = 0
    For j As Integer = 0 To vocabSize-1
        ws(0)->wp(j) = Exp(ws(0)->wlog(j) - mx) : sm += ws(0)->wp(j)
    Next
    Dim As Double r = Rnd * sm
    Dim As Double acc = 0
    For j As Integer = 0 To vocabSize-1
        acc += ws(0)->wp(j)
        If r <= acc Then Return j
    Next
    Return vocabSize-1
End Function

Function encodeInto(ctx() As Integer, ctxLen As Integer, s As String) As Integer
    Dim As Integer spaceId = ch2id(Asc(" "))
    If spaceId < 0 Then spaceId = 0
    Dim As Integer n = ctxLen
    For i As Integer = 0 To Len(s)-1
        Dim As Integer id = ch2id(s[i])
        If id < 0 Then id = spaceId
        ctx(n) = id : n += 1
    Next
    Return n
End Function

'' =====================================================================
''  Multi-threaded training
'' =====================================================================
Dim Shared As Integer batchStart(BATCH-1)   '' random window starts for this step
Dim Shared As Double  threadLoss(NTHREAD-1)

Sub workerProc(ud As Any Ptr)
    Dim As Integer tid = Cast(Integer, ud)
    '' zero this thread's gradient buffers
    For i As Integer = 0 To nparams-1
        zeroPtr(params(i).gradT(tid), params(i).n)
    Next
    Dim As Integer tok(BLOCK-1), tgt(BLOCK-1)
    Dim As Double lsum = 0
    Dim As Integer b = tid
    While b < BATCH
        Dim As Integer r = batchStart(b)
        For t As Integer = 0 To BLOCK-1
            tok(t) = dataArr(r + t)
            tgt(t) = dataArr(r + t + 1)
        Next
        lsum += backward(*ws(tid), tid, tok(), tgt(), BLOCK)
        b += NTHREAD
    Wend
    threadLoss(tid) = lsum
End Sub

Sub train(nSteps As Integer)
    loadCorpus()
    buildModel()
    ensureWorkspaces(NTHREAD)

    Dim As Integer total = 0
    For i As Integer = 0 To nparams-1 : total += params(i).n : Next
    Print "model: "; total; " parameters ("; NLAYER; " layers, "; NEMB; " dim, "; _
          NHEAD; " heads, ctx "; BLOCK; ")"
    Print "training "; nSteps; " steps, batch "; BATCH; " on "; NTHREAD; " threads"
    Print

    Dim As Double running = 0
    Dim As Integer rcount = 0
    Dim As Double t0 = Timer
    Dim As Any Ptr th(NTHREAD-1)
    Dim As Single binv = 1.0 / BATCH

    For st As Integer = 1 To nSteps
        Dim As Single lr
        If st <= WARMUP Then
            lr = LR_MAX * st / WARMUP
        Else
            Dim As Single prog = (st - WARMUP) / (nSteps - WARMUP + 1)
            lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + Cos(3.14159265 * prog))
        End If

        '' choose batch windows (deterministic, main thread only)
        For b As Integer = 0 To BATCH-1
            batchStart(b) = Int(Rnd * (nTok - BLOCK - 1))
        Next

        '' run workers
        For tid As Integer = 0 To NTHREAD-1
            th(tid) = ThreadCreate(@workerProc, Cast(Any Ptr, tid))
        Next
        For tid As Integer = 0 To NTHREAD-1
            ThreadWait(th(tid))
        Next

        '' reduce per-thread gradients -> params.grad, averaged over batch
        For i As Integer = 0 To nparams-1
            Dim As Single Ptr g = params(i).grad
            zeroPtr(g, params(i).n)
            For tid As Integer = 0 To NTHREAD-1
                Dim As Single Ptr gt = params(i).gradT(tid)
                For k As Integer = 0 To params(i).n-1 : g[k] += gt[k] : Next
            Next
            For k As Integer = 0 To params(i).n-1 : g[k] *= binv : Next
        Next

        Dim As Double batchLoss = 0
        For tid As Integer = 0 To NTHREAD-1 : batchLoss += threadLoss(tid) : Next

        adamStep(st, lr)

        running += batchLoss * binv
        rcount += 1

        If st Mod 25 = 0 Then
            Dim As Double avg = running / rcount
            Dim As Double dt = Timer - t0
            Dim As Double sps = st / dt
            Print Using "step ##### / #####   loss ##.####   lr #.#####   (##.## steps/s)"; _
                  st; nSteps; avg; lr; sps
            running = 0 : rcount = 0
        End If

        If st Mod 500 = 0 Then
            saveModel()
            Print "  sample: ";
            Dim As Integer ctx(4095), n = 0
            n = encodeInto(ctx(), n, "User: hello" + Chr(10) + "Bot:")
            For g As Integer = 1 To 120
                Dim As Integer id = sampleNext(ctx(), n, 0.8)
                ctx(n) = id : n += 1
                Dim As Integer c = id2ch(id)
                If c = 10 Then Exit For
                Print Chr(c);
                If n >= 4090 Then Exit For
            Next
            Print : Print
        End If
    Next

    saveModel()
    Print "done."
End Sub

'' =====================================================================
''  Interactive chat
'' =====================================================================
Sub chat()
    If loadModel() = 0 Then
        Print "No trained model found. Run:  atlas train"
        End 1
    End If
    ensureWorkspaces(1)
    Print "Atlas ("; vocabSize; " symbols). Type a message; '/quit' to exit."
    Print "----------------------------------------------------------------"

    '' Input handling.  FreeBASIC's runtime puts the terminal in raw mode
    '' (echo + canonical OFF) at startup so its own input functions can echo
    '' and line-edit manually.  So:
    ''   * interactive terminal -> use FB's Line Input (it echoes correctly);
    ''   * piped/redirected      -> use the C runtime's fgets (Line Input would
    ''                              hang on the ESC[6n handshake with no tty).
    '' isatty(0) tells the two apart.
    Dim As Integer interactive = (isatty(0) <> 0)
    Dim As Integer ctx(8191), n = 0
    Dim buf As ZString * 8192
    Do
        Dim As String userLine
        If interactive Then
            Line Input "You: ", userLine
        Else
            Print "You: ";
            fflush(stdout)
            If fgets(@buf, 8192, stdin) = 0 Then Exit Do   '' NULL => end of input
            userLine = *Cast(ZString Ptr, @buf)
        End If
        userLine = RTrim(userLine, Any Chr(13) & Chr(10))   '' strip trailing CR/LF
        If userLine = "/quit" OrElse userLine = "/exit" Then Exit Do
        If Len(userLine) = 0 Then Continue Do

        n = encodeInto(ctx(), n, "User: " + userLine + Chr(10) + "Bot:")

        Print "Bot:";
        Dim As Integer produced = 0
        For g As Integer = 1 To 400
            Dim As Integer id = sampleNext(ctx(), n, 0.8)
            ctx(n) = id : n += 1
            Dim As Integer c = id2ch(id)
            If c = 10 Then
                If produced > 0 Then Exit For
            Else
                Print Chr(c);
                produced += 1
            End If
            If n >= 8000 Then
                Dim As Integer keep = 4000
                For i As Integer = 0 To keep-1 : ctx(i) = ctx(n - keep + i) : Next
                n = keep
            End If
        Next
        ctx(n) = ch2id(Asc(Chr(10))) : n += 1
        Print
    Loop
    Print
    Print "goodbye."
End Sub

'' =====================================================================
''  Entry point
'' =====================================================================
Randomize 1234

Dim As String mode = LCase(Command(1))

Select Case mode
Case "train"
    Dim As Integer steps = 2000
    If Len(Command(2)) > 0 Then steps = Val(Command(2))
    train(steps)
Case "chat"
    chat()
Case "help", "-h", "--help"
    Print "atlas train [steps]   - train from "; CORPUS_FILE; ", save "; MODEL_FILE
    Print "atlas chat            - load "; MODEL_FILE; " and chat"
Case Else
    Dim As Integer f = FreeFile
    If Open(MODEL_FILE For Binary Access Read As #f) = 0 Then
        Close #f
        chat()
    Else
        Print "No model yet - training a fresh one."
        Print
        train(2000)
    End If
End Select
