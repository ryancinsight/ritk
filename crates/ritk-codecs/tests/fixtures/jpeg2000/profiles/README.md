# JPEG 2000 packet and precision fixtures

These raw codestreams contain generated 35 × 33 test patterns. OpenJPEG 2.5.3 and OpenJPH 0.21.2 produced and independently decoded the images. OpenJPH 0.33.0 produced the signed 30-bit cases. All files are free of patient data.

| Files | Construction and reference |
| --- | --- |
| `classic-{order}` | OpenJPEG with `-n 3 -p {order} -r 7,3,1 -b 8,8 -c [16,16],[8,8],[8,8] -SOP -EPH` |
| `classic-tileparts`, `classic-poc` | OpenJPEG with `-p RLCP -t 18,16 -TP R -r 7,1` or `-POC T1=0,0,1,3,1,RLCP -r 1`, both with `-n 3` |
| `classic-ppm`, `classic-ppt` | Move the inline LRCP packet headers through EPH into one PPM or PPT segment and update Psot. Each decoded OpenJPEG output equals the original. |
| `classic-tile-override`, `classic-component-override` | Copy COD and QCD into the tile and change the main progression to CPRL, or write component-zero COC and QCC from the main defaults. OpenJPEG output equals the original for both. |
| `ht-{order}`, `ht-tiles` | OpenJPH with three decompositions and reversible coding. Tiled CPRL uses 18 × 17 tiles. |
| `ht-rpcl-parts` | OpenJPH RPCL with `-tileparts R -tlm_marker true -precincts {8,8} -block_size {8,8}` |
| `ht-irreversible` | OpenJPH irreversible LRCP with three decompositions and `-qstep 0.001`. The `.u16le` file contains the independent decoder output, one unsigned little-endian word per pixel. |
| `*-color-{rgb,rct}` | Three 8-bit components with color transform disabled or enabled. Two decompositions, reversible transform. |
| `ht-color-ict` | OpenJPH irreversible three-component transform, two decompositions, normalized quantization step 0.001. The `.rgb` file is its decoded 8-bit interleaved RGB output. |
| `ht-deep-{basic,advanced}` | OpenJPH raw signed 30-bit input, zero decompositions. The advanced case uses 16 × 16 precincts and TLM. OpenJPH reproduced all 1,155 signed input values. |

For the 12-bit inputs, the sample at zero-based (x,y) is `(31*x + 73*y + 17*(x XOR y)) % 4096`. For RGB channel c it is `(11*x + 17*y + 57*c + (c+1)*(x XOR y)) % 256`. For the signed 30-bit input the magnitude is `2^27 + 35*y + x`, with a negative sign on odd x+y. The 38-bit tests reuse the reversible coefficients and change Ssiz to unsigned 38, which adds the known DC level shift of 2^37 to signed data.

The reverse-direction test sends 147 × 131 unsigned 16-bit pixels to RITK. OpenJPH 0.33.0 decoded both lossless writers exactly and the irreversible writer within one stored unit. The input is `(371*x) XOR (197*y)`.
