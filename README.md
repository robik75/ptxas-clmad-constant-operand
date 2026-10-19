# ptxas drops half of a folded clmad (CUDA 13.4.92, sm_120)

ptxas in CUDA 13.4.92 (13.4 Update 2) computes a wrong 128-bit carry-less product on sm_120 at `-O2` and `-O3`. It happens when ptxas propagates the constant one into a `clmad.lo.u64` and folds that instruction into a 64-bit copy of its other operand: it emits only the low 32-bit half of the copy, and the high half reads a register that is never written. The driver's JIT compiler (615.71.09) produces the same wrong result from the PTX. `-Xptxas -O1` and `-O0` are correct.

The kernel is executed and every result is checked on the host against a bitwise carry-less multiply. The SASS listings come from the toolkit's own `cuobjdump` (13.4.92), which decodes `CLMAD`.

## Reproduce

```
./run.sh
```

`run.sh` builds `repro.cu` for `sm_120` at `-Xptxas -O3`, `-O2`, `-O1` and `-O0`, and once as PTX only (`-gencode arch=compute_120,code=compute_120`) so that the driver JIT-compiles it. It keeps the PTX and the SASS listings in `out/` and runs every build. Each run exits 0 when every step matches the host, 1 when one doesn't and 2 on a CUDA error.

`repro.cu` multiplies 128-bit values `(lo, hi)` carry-lessly. `clmul` is one 64x64 product (`clmad.lo.u64` and `clmad.hi.u64` with a zero addend), and `mul` forms the 256-bit product from three of them (Karatsuba), then folds it to 128 bits. One thread computes the running product `e = 1, e = mul(e, a[t])` for `t < 14` and stores every step. When one factor is one, the product's high half is zero, so the first step must equal `a[0]`.

## Result

```
nvcc: Cuda compilation tools, release 13.4, V13.4.92
cuobjdump: Cuda compilation tools, release 13.4, V13.4.92
NVIDIA RTX PRO 6000 Blackwell Server Edition, 615.71.09, 12.0
== ptxas -O3
step  0: got 1d4713583167fa23 7d49bf6b24b0c5ec, want 8b5868123167fa23 7d49bf6b24b0c5ec
14 of 14 steps wrong
exit 1
== ptxas -O2
step  0: got 0d4e1b4b3167fa23 7d49bf6b24b0c5ec, want 8b5868123167fa23 7d49bf6b24b0c5ec
14 of 14 steps wrong
exit 1
== ptxas -O1
0 of 14 steps wrong
exit 0
== ptxas -O0
0 of 14 steps wrong
exit 0
== PTX only, JIT-compiled by the driver
step  0: got 0d4413583167fa23 7d49bf6b24b0c5ec, want 8b5868123167fa23 7d49bf6b24b0c5ec
14 of 14 steps wrong
exit 1
```

Values print as the high 64-bit word, then the low one; the output above omits steps 1 to 13, which are wrong because step 0 is. In step 0 only the top 32 bits of the high word are wrong, and they change from run to run of the same binary.

The trip count matters: with the default `-O3` the first step is wrong for 14, 15, 18 and 19 steps and right for 10, 12, 13, 16, 17, 20 and 24 (`./run.sh -DN=16`, for example). Computing the same 256-bit product schoolbook, with four 64x64 products instead of three, gave right results at every trip count we tried (8, 10, 12 to 20, 24 and 32).

## What the SASS shows

In the PTX (`out/repro.ptx`) the loop runs over two steps per iteration, and `e = (1, 0)` enters it in registers; every product is a `clmad`. ptxas unrolls the loop fully, propagates the one into the first step and folds that step's `clmad`s away. With `b = a[0]`, the first step is `mul(1, b)`: its middle product `clmad.lo(1, b.lo ^ b.hi, 0)` is just `b.lo ^ b.hi`, and the result is `(b.lo, b.hi)`. The `-O3` SASS (`out/repro_O3.sass`):

```
/*0030*/  LDG.E.64 R14, desc[UR4][R2.64] ;            b.lo in R15:R14
/*0040*/  LDG.E.64 R4, desc[UR4][R2.64+0x8] ;         b.hi in R5:R4
/*0050*/  LOP3.LUT R4, R4, R14, RZ, 0x3c, !PT ;       R5:R4 = b.lo ^ b.hi, the folded clmad.lo ...
/*0060*/  LOP3.LUT R5, R5, R15, RZ, 0x3c, !PT ;
/*0070*/  LOP3.LUT R16, R4, RZ, RZ, 0x3c, !PT ;       ... copied to R17:R16, but only R16 = R4
/*0080*/  LDC.64 R4, c[0x0][0x388] ;                  R5:R4 overwritten
/*0090*/  LOP3.LUT R18, R16, R14, RZ, 0x3c, !PT ;     R18 = R16 ^ R14, the high word's low half
/*00a0*/  LOP3.LUT R19, R17, R15, RZ, 0x3c, !PT ;     R19 = R17 ^ R15, but R17 was never written
/*00b0*/  STG.E.64 desc[UR4][R4.64], R14 ;            stores b.lo
/*00c0*/  STG.E.64 desc[UR4][R4.64+0x8], R18 ;        stores R19:R18 as the high word
```

The copy at `/*0070*/` has no high-half counterpart (`R17 = R5`), so `/*00a0*/` reads `R17` before any instruction writes it, which is why the wrong bits vary between runs. Of the 84 `clmad`s the 14 steps need, the `-O3` SASS has 78: the first step's six are the ones folded.

## Environment

- CUDA 13.4.92 (13.4 Update 2; the toolkit 13.4.3 metapackage of 2026-10-14 pins the same 13.4.92 compiler): `nvcc`, `ptxas` and `cuobjdump`
- NVIDIA RTX PRO 6000 Blackwell Server Edition (compute capability 12.0), driver 615.71.09
- Linux x86_64

## Workaround

Don't let ptxas see a constant multiplicand: start a running product from its first factor instead of from one, and handle multiplication by one or zero without a `clmad`.
