# AETHER x86-64 assembly (Microsoft x64 ABI)
# Emitted by aetherc; comments here are debug-only and do not
# come from any .aether source — those were stripped at lex time.

.section .rdata,"dr"
.LF_main_0:
    .byte 0x00
    .byte 0x00
    .byte 0x80
    .byte 0x3f
.LF_main_1:
    .byte 0xcd
    .byte 0xcc
    .byte 0xcc
    .byte 0x3d

.section .text
.globl main

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $128, %rsp
    addq $0, %rsp
    callq aether_dev_init
    movq $128, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rcx
    addq $16, %rsp
    callq aether_alloc_f32
    movq %rax, -16(%rbp)
    movq $64, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rcx
    addq $16, %rsp
    callq aether_alloc_f32
    movq %rax, -24(%rbp)
    movq -16(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $128, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movss .LF_main_0(%rip), %xmm0
    subq $16, %rsp
    movss %xmm0, (%rsp)
    movq $1, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %r9
    movss 16(%rsp), %xmm0
    movss %xmm0, %xmm2
    movq 32(%rsp), %rax
    movq %rax, %rdx
    movq 48(%rsp), %rax
    movq %rax, %rcx
    addq $64, %rsp
    callq aether_init_normal_f32
    movq -24(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $64, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movss .LF_main_1(%rip), %xmm0
    subq $16, %rsp
    movss %xmm0, (%rsp)
    movq $7, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %r9
    movss 16(%rsp), %xmm0
    movss %xmm0, %xmm2
    movq 32(%rsp), %rax
    movq %rax, %rdx
    movq 48(%rsp), %rax
    movq %rax, %rcx
    addq $64, %rsp
    callq aether_init_normal_f32
    movq $128, %rax
    movq %rax, %rcx
    callq aether_dev_alloc_f32
    movq %rax, -32(%rbp)
    movq $64, %rax
    movq %rax, %rcx
    callq aether_dev_alloc_f32
    movq %rax, -40(%rbp)
    movq $32, %rax
    movq %rax, %rcx
    callq aether_dev_alloc_f32
    movq %rax, -48(%rbp)
    movq -16(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq -32(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $128, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %r8
    movq 16(%rsp), %rax
    movq %rax, %rdx
    movq 32(%rsp), %rax
    movq %rax, %rcx
    addq $48, %rsp
    callq aether_dev_h2d_f32
    movq -24(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq -40(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $64, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %r8
    movq 16(%rsp), %rax
    movq %rax, %rdx
    movq 32(%rsp), %rax
    movq %rax, %rcx
    addq $48, %rsp
    callq aether_dev_h2d_f32
    movq -32(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq -40(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq -48(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %r8
    movq 16(%rsp), %rax
    movq %rax, %rdx
    movq 32(%rsp), %rax
    movq %rax, %rcx
    addq $48, %rsp
    callq aether_forward__M8__K16__N4
    movq $128, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rcx
    addq $16, %rsp
    callq aether_alloc_f32
    movq %rax, -56(%rbp)
    movq $64, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rcx
    addq $16, %rsp
    callq aether_alloc_f32
    movq %rax, -64(%rbp)
    movq -56(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $128, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movss .LF_main_0(%rip), %xmm0
    subq $16, %rsp
    movss %xmm0, (%rsp)
    movq $3, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %r9
    movss 16(%rsp), %xmm0
    movss %xmm0, %xmm2
    movq 32(%rsp), %rax
    movq %rax, %rdx
    movq 48(%rsp), %rax
    movq %rax, %rcx
    addq $64, %rsp
    callq aether_init_normal_f32
    movq -64(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $64, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movss .LF_main_1(%rip), %xmm0
    subq $16, %rsp
    movss %xmm0, (%rsp)
    movq $11, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %r9
    movss 16(%rsp), %xmm0
    movss %xmm0, %xmm2
    movq 32(%rsp), %rax
    movq %rax, %rdx
    movq 48(%rsp), %rax
    movq %rax, %rcx
    addq $64, %rsp
    callq aether_init_normal_f32
    movq $128, %rax
    movq %rax, %rcx
    callq aether_dev_alloc_f32
    movq %rax, -72(%rbp)
    movq $64, %rax
    movq %rax, %rcx
    callq aether_dev_alloc_f32
    movq %rax, -80(%rbp)
    movq $8, %rax
    movq %rax, %rcx
    callq aether_dev_alloc_f32
    movq %rax, -88(%rbp)
    movq -56(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq -72(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $128, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %r8
    movq 16(%rsp), %rax
    movq %rax, %rdx
    movq 32(%rsp), %rax
    movq %rax, %rcx
    addq $48, %rsp
    callq aether_dev_h2d_f32
    movq -64(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq -80(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $64, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %r8
    movq 16(%rsp), %rax
    movq %rax, %rdx
    movq 32(%rsp), %rax
    movq %rax, %rcx
    addq $48, %rsp
    callq aether_dev_h2d_f32
    movq -72(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq -80(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq -88(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %r8
    movq 16(%rsp), %rax
    movq %rax, %rdx
    movq 32(%rsp), %rax
    movq %rax, %rcx
    addq $48, %rsp
    callq aether_forward__M4__K32__N2
    addq $0, %rsp
    callq aether_dev_sync
    movq -16(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $128, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rdx
    movq 16(%rsp), %rax
    movq %rax, %rcx
    addq $32, %rsp
    callq aether_free_f32
    movq -24(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $64, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rdx
    movq 16(%rsp), %rax
    movq %rax, %rcx
    addq $32, %rsp
    callq aether_free_f32
    movq -56(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $128, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rdx
    movq 16(%rsp), %rax
    movq %rax, %rcx
    addq $32, %rsp
    callq aether_free_f32
    movq -64(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $64, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rdx
    movq 16(%rsp), %rax
    movq %rax, %rcx
    addq $32, %rsp
    callq aether_free_f32
    movq $0, %rax
    movq %rax, -8(%rbp)
    movq -88(%rbp), %rcx
    callq aether_dev_free_f32
    movq -80(%rbp), %rcx
    callq aether_dev_free_f32
    movq -72(%rbp), %rcx
    callq aether_dev_free_f32
    movq -48(%rbp), %rcx
    callq aether_dev_free_f32
    movq -40(%rbp), %rcx
    callq aether_dev_free_f32
    movq -32(%rbp), %rcx
    callq aether_dev_free_f32
    movq -8(%rbp), %rax
    addq $128, %rsp
    popq %rbp
    ret

aether_forward__M4__K32__N2:
    pushq %rbp
    movq %rsp, %rbp
    subq $80, %rsp
    movq %rcx, -8(%rbp)
    movq %rdx, -16(%rbp)
    movq %r8, -24(%rbp)
    movq -8(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq -16(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq -24(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $4, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $32, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $2, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, 136(%rsp)
    movq 16(%rsp), %rax
    movq %rax, 128(%rsp)
    movq 32(%rsp), %rax
    movq %rax, %r9
    movq 48(%rsp), %rax
    movq %rax, %r8
    movq 64(%rsp), %rax
    movq %rax, %rdx
    movq 80(%rsp), %rax
    movq %rax, %rcx
    addq $96, %rsp
    callq aether_op_matmul_f32_cuda
    movq $0, %rax
    addq $80, %rsp
    popq %rbp
    ret

aether_forward__M8__K16__N4:
    pushq %rbp
    movq %rsp, %rbp
    subq $80, %rsp
    movq %rcx, -8(%rbp)
    movq %rdx, -16(%rbp)
    movq %r8, -24(%rbp)
    movq -8(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq -16(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq -24(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $8, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $16, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $4, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, 136(%rsp)
    movq 16(%rsp), %rax
    movq %rax, 128(%rsp)
    movq 32(%rsp), %rax
    movq %rax, %r9
    movq 48(%rsp), %rax
    movq %rax, %r8
    movq 64(%rsp), %rax
    movq %rax, %rdx
    movq 80(%rsp), %rax
    movq %rax, %rcx
    addq $96, %rsp
    callq aether_op_matmul_f32_cuda
    movq $0, %rax
    addq $80, %rsp
    popq %rbp
    ret

