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
.LF_main_2:
    .byte 0x00
    .byte 0x00
    .byte 0x00
    .byte 0x00
.LF_main_3:
    .byte 0xcd
    .byte 0xcc
    .byte 0x4c
    .byte 0x3d
.LF_main_4:
    .byte 0x66
    .byte 0x66
    .byte 0x66
    .byte 0x3f
.LF_main_5:
    .byte 0x77
    .byte 0xbe
    .byte 0x7f
    .byte 0x3f
.LF_main_6:
    .byte 0x77
    .byte 0xcc
    .byte 0x2b
    .byte 0x32

.section .text
.globl main

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $224, %rsp
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
    movq $8, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rcx
    addq $16, %rsp
    callq aether_alloc_i32
    movq %rax, -32(%rbp)
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
    movq -32(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $8, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $4, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $13, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %r9
    movq 16(%rsp), %rax
    movq %rax, %r8
    movq 32(%rsp), %rax
    movq %rax, %rdx
    movq 48(%rsp), %rax
    movq %rax, %rcx
    addq $64, %rsp
    callq aether_fill_labels_i32
    movq $128, %rax
    movq %rax, %rcx
    callq aether_dev_alloc_f32
    movq %rax, -40(%rbp)
    movq $64, %rax
    movq %rax, %rcx
    callq aether_dev_alloc_f32
    movq %rax, -48(%rbp)
    movq $32, %rax
    movq %rax, %rcx
    callq aether_dev_alloc_f32
    movq %rax, -56(%rbp)
    movq $32, %rax
    movq %rax, %rcx
    callq aether_dev_alloc_f32
    movq %rax, -64(%rbp)
    movq $32, %rax
    movq %rax, %rcx
    callq aether_dev_alloc_f32
    movq %rax, -72(%rbp)
    movq $64, %rax
    movq %rax, %rcx
    callq aether_dev_alloc_f32
    movq %rax, -80(%rbp)
    movq $64, %rax
    movq %rax, %rcx
    callq aether_dev_alloc_f32
    movq %rax, -88(%rbp)
    movq $64, %rax
    movq %rax, %rcx
    callq aether_dev_alloc_f32
    movq %rax, -96(%rbp)
    movq $8, %rax
    movq %rax, %rcx
    callq aether_dev_alloc_i32
    movq %rax, -104(%rbp)
    movq -16(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq -40(%rbp), %rax
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
    movq -48(%rbp), %rax
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
    movq -104(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $8, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %r8
    movq 16(%rsp), %rax
    movq %rax, %rdx
    movq 32(%rsp), %rax
    movq %rax, %rcx
    addq $48, %rsp
    callq aether_dev_h2d_i32
    movss .LF_main_2(%rip), %xmm0
    movss %xmm0, -112(%rbp)
    movss .LF_main_2(%rip), %xmm0
    movss %xmm0, -120(%rbp)
    movq $1, %rax
    movq %rax, -128(%rbp)
.L_main_while_top_0:
    movq -128(%rbp), %rax
    pushq %rax
    movq $50, %rax
    popq %r10
    xchgq %rax, %r10
    cmpq %r10, %rax
    setle %al
    movzbl %al, %eax
    testq %rax, %rax
    je .L_main_while_end_1
    movq -40(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq -48(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq -56(%rbp), %rax
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
    movq -56(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq -104(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq -64(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $8, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $4, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, 112(%rsp)
    movq 16(%rsp), %rax
    movq %rax, %r9
    movq 32(%rsp), %rax
    movq %rax, %r8
    movq 48(%rsp), %rax
    movq %rax, %rdx
    movq 64(%rsp), %rax
    movq %rax, %rcx
    addq $80, %rsp
    callq aether_op_cross_entropy_f32_cuda
    movss %xmm0, -136(%rbp)
    movq -128(%rbp), %rax
    pushq %rax
    movq $1, %rax
    popq %r10
    xchgq %rax, %r10
    cmpq %r10, %rax
    sete %al
    movzbl %al, %eax
    testq %rax, %rax
    je .L_main_else_2
    movss -136(%rbp), %xmm0
    movss %xmm0, -112(%rbp)
    jmp .L_main_endif_3
.L_main_else_2:
.L_main_endif_3:
    movss -136(%rbp), %xmm0
    movss %xmm0, -120(%rbp)
    movq -64(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq -104(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq -72(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $8, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $4, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, 112(%rsp)
    movq 16(%rsp), %rax
    movq %rax, %r9
    movq 32(%rsp), %rax
    movq %rax, %r8
    movq 48(%rsp), %rax
    movq %rax, %rdx
    movq 64(%rsp), %rax
    movq %rax, %rcx
    addq $80, %rsp
    callq aether_op_cross_entropy_backward_f32_cuda
    movq -40(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq -72(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq -80(%rbp), %rax
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
    callq aether_op_matmul_backward_rhs_f32_cuda
    movq -48(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq -80(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq -88(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq -96(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movss .LF_main_3(%rip), %xmm0
    subq $16, %rsp
    movss %xmm0, (%rsp)
    movss .LF_main_4(%rip), %xmm0
    subq $16, %rsp
    movss %xmm0, (%rsp)
    movss .LF_main_5(%rip), %xmm0
    subq $16, %rsp
    movss %xmm0, (%rsp)
    movss .LF_main_6(%rip), %xmm0
    subq $16, %rsp
    movss %xmm0, (%rsp)
    movss .LF_main_2(%rip), %xmm0
    subq $16, %rsp
    movss %xmm0, (%rsp)
    movq -128(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $64, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, 256(%rsp)
    movq 16(%rsp), %rax
    movq %rax, 248(%rsp)
    movss 32(%rsp), %xmm0
    movss %xmm0, 240(%rsp)
    movss 48(%rsp), %xmm0
    movss %xmm0, 232(%rsp)
    movss 64(%rsp), %xmm0
    movss %xmm0, 224(%rsp)
    movss 80(%rsp), %xmm0
    movss %xmm0, 216(%rsp)
    movss 96(%rsp), %xmm0
    movss %xmm0, 208(%rsp)
    movq 112(%rsp), %rax
    movq %rax, %r9
    movq 128(%rsp), %rax
    movq %rax, %r8
    movq 144(%rsp), %rax
    movq %rax, %rdx
    movq 160(%rsp), %rax
    movq %rax, %rcx
    addq $176, %rsp
    callq aether_op_adamw_step_f32_cuda
    movq -128(%rbp), %rax
    pushq %rax
    movq $1, %rax
    popq %r10
    xchgq %rax, %r10
    addq %r10, %rax
    movq %rax, -128(%rbp)
    jmp .L_main_while_top_0
.L_main_while_end_1:
    xorl %eax, %eax
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
    movq -32(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $8, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rdx
    movq 16(%rsp), %rax
    movq %rax, %rcx
    addq $32, %rsp
    callq aether_free_i32
    movss -120(%rbp), %xmm0
    subq $16, %rsp
    movss %xmm0, (%rsp)
    movss -112(%rbp), %xmm0
    movss %xmm0, %xmm1
    movss (%rsp), %xmm0
    addq $16, %rsp
    ucomiss %xmm1, %xmm0
    setb %al
    movzbl %al, %eax
    testq %rax, %rax
    je .L_main_else_4
    movq $0, %rax
    jmp .L_main_endif_5
.L_main_else_4:
    movq $1, %rax
.L_main_endif_5:
    movq %rax, -8(%rbp)
    movq -104(%rbp), %rcx
    callq aether_dev_free_i32
    movq -96(%rbp), %rcx
    callq aether_dev_free_f32
    movq -88(%rbp), %rcx
    callq aether_dev_free_f32
    movq -80(%rbp), %rcx
    callq aether_dev_free_f32
    movq -72(%rbp), %rcx
    callq aether_dev_free_f32
    movq -64(%rbp), %rcx
    callq aether_dev_free_f32
    movq -56(%rbp), %rcx
    callq aether_dev_free_f32
    movq -48(%rbp), %rcx
    callq aether_dev_free_f32
    movq -40(%rbp), %rcx
    callq aether_dev_free_f32
    movq -8(%rbp), %rax
    addq $224, %rsp
    popq %rbp
    ret

