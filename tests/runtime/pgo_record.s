# AETHER x86-64 assembly (Microsoft x64 ABI)
# Emitted by aetherc; comments here are debug-only and do not
# come from any .aether source — those were stripped at lex time.

.section .rdata,"dr"
.LF_main_0:
    .byte 0x00
    .byte 0x00
    .byte 0x00
    .byte 0x00
.LF_main_1:
    .byte 0x00
    .byte 0x00
    .byte 0x00
    .byte 0x3f
.LF_main_2:
    .byte 0x00
    .byte 0x00
    .byte 0x80
    .byte 0x3f

.section .text
.globl main

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $80, %rsp
    addq $0, %rsp
    callq aether_pgo_reset
    movq $1, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rcx
    addq $16, %rsp
    callq aether_pgo_call_count
    pushq %rax
    movq $0, %rax
    popq %r10
    xchgq %rax, %r10
    cmpq %r10, %rax
    setne %al
    movzbl %al, %eax
    testq %rax, %rax
    je .L_main_else_0
    movq $1, %rax
    addq $48, %rsp
    popq %rbp
    ret
    jmp .L_main_endif_1
.L_main_else_0:
.L_main_endif_1:
    movq $1, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rcx
    addq $16, %rsp
    callq aether_pgo_branch_freq
    movss %xmm0, -16(%rbp)
    movss -16(%rbp), %xmm0
    subq $16, %rsp
    movss %xmm0, (%rsp)
    movss .LF_main_0(%rip), %xmm0
    movss %xmm0, %xmm1
    movss (%rsp), %xmm0
    addq $16, %rsp
    ucomiss %xmm1, %xmm0
    setne %al
    movzbl %al, %eax
    testq %rax, %rax
    je .L_main_else_2
    movq $2, %rax
    addq $48, %rsp
    popq %rbp
    ret
    jmp .L_main_endif_3
.L_main_else_2:
.L_main_endif_3:
    movq $1, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $1, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rdx
    movq 16(%rsp), %rax
    movq %rax, %rcx
    addq $32, %rsp
    callq aether_pgo_record_branch
    movq $1, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $0, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rdx
    movq 16(%rsp), %rax
    movq %rax, %rcx
    addq $32, %rsp
    callq aether_pgo_record_branch
    movq $1, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $1, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rdx
    movq 16(%rsp), %rax
    movq %rax, %rcx
    addq $32, %rsp
    callq aether_pgo_record_branch
    movq $1, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $0, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rdx
    movq 16(%rsp), %rax
    movq %rax, %rcx
    addq $32, %rsp
    callq aether_pgo_record_branch
    movq $1, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $1, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rdx
    movq 16(%rsp), %rax
    movq %rax, %rcx
    addq $32, %rsp
    callq aether_pgo_record_branch
    movq $1, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $0, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rdx
    movq 16(%rsp), %rax
    movq %rax, %rcx
    addq $32, %rsp
    callq aether_pgo_record_branch
    movq $2, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $1, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rdx
    movq 16(%rsp), %rax
    movq %rax, %rcx
    addq $32, %rsp
    callq aether_pgo_record_branch
    movq $2, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $1, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rdx
    movq 16(%rsp), %rax
    movq %rax, %rcx
    addq $32, %rsp
    callq aether_pgo_record_branch
    movq $2, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $1, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rdx
    movq 16(%rsp), %rax
    movq %rax, %rcx
    addq $32, %rsp
    callq aether_pgo_record_branch
    movq $2, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $1, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rdx
    movq 16(%rsp), %rax
    movq %rax, %rcx
    addq $32, %rsp
    callq aether_pgo_record_branch
    movq $1, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rcx
    addq $16, %rsp
    callq aether_pgo_branch_freq
    movss %xmm0, -24(%rbp)
    movss -24(%rbp), %xmm0
    subq $16, %rsp
    movss %xmm0, (%rsp)
    movss .LF_main_1(%rip), %xmm0
    movss %xmm0, %xmm1
    movss (%rsp), %xmm0
    addq $16, %rsp
    ucomiss %xmm1, %xmm0
    setne %al
    movzbl %al, %eax
    testq %rax, %rax
    je .L_main_else_4
    movq $3, %rax
    addq $64, %rsp
    popq %rbp
    ret
    jmp .L_main_endif_5
.L_main_else_4:
.L_main_endif_5:
    movq $2, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rcx
    addq $16, %rsp
    callq aether_pgo_branch_freq
    movss %xmm0, -32(%rbp)
    movss -32(%rbp), %xmm0
    subq $16, %rsp
    movss %xmm0, (%rsp)
    movss .LF_main_2(%rip), %xmm0
    movss %xmm0, %xmm1
    movss (%rsp), %xmm0
    addq $16, %rsp
    ucomiss %xmm1, %xmm0
    setne %al
    movzbl %al, %eax
    testq %rax, %rax
    je .L_main_else_6
    movq $4, %rax
    addq $64, %rsp
    popq %rbp
    ret
    jmp .L_main_endif_7
.L_main_else_6:
.L_main_endif_7:
    movq $99, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rcx
    addq $16, %rsp
    callq aether_pgo_branch_freq
    movss %xmm0, -40(%rbp)
    movss -40(%rbp), %xmm0
    subq $16, %rsp
    movss %xmm0, (%rsp)
    movss .LF_main_0(%rip), %xmm0
    movss %xmm0, %xmm1
    movss (%rsp), %xmm0
    addq $16, %rsp
    ucomiss %xmm1, %xmm0
    setne %al
    movzbl %al, %eax
    testq %rax, %rax
    je .L_main_else_8
    movq $5, %rax
    addq $80, %rsp
    popq %rbp
    ret
    jmp .L_main_endif_9
.L_main_else_8:
.L_main_endif_9:
    movq $10, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rcx
    addq $16, %rsp
    callq aether_pgo_record_call
    movq $10, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rcx
    addq $16, %rsp
    callq aether_pgo_record_call
    movq $10, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rcx
    addq $16, %rsp
    callq aether_pgo_record_call
    movq $20, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rcx
    addq $16, %rsp
    callq aether_pgo_record_call
    movq $10, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rcx
    addq $16, %rsp
    callq aether_pgo_call_count
    pushq %rax
    movq $3, %rax
    popq %r10
    xchgq %rax, %r10
    cmpq %r10, %rax
    setne %al
    movzbl %al, %eax
    testq %rax, %rax
    je .L_main_else_10
    movq $6, %rax
    addq $80, %rsp
    popq %rbp
    ret
    jmp .L_main_endif_11
.L_main_else_10:
.L_main_endif_11:
    movq $20, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rcx
    addq $16, %rsp
    callq aether_pgo_call_count
    pushq %rax
    movq $1, %rax
    popq %r10
    xchgq %rax, %r10
    cmpq %r10, %rax
    setne %al
    movzbl %al, %eax
    testq %rax, %rax
    je .L_main_else_12
    movq $7, %rax
    addq $80, %rsp
    popq %rbp
    ret
    jmp .L_main_endif_13
.L_main_else_12:
.L_main_endif_13:
    movq $999, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rcx
    addq $16, %rsp
    callq aether_pgo_call_count
    pushq %rax
    movq $0, %rax
    popq %r10
    xchgq %rax, %r10
    cmpq %r10, %rax
    setne %al
    movzbl %al, %eax
    testq %rax, %rax
    je .L_main_else_14
    movq $8, %rax
    addq $80, %rsp
    popq %rbp
    ret
    jmp .L_main_endif_15
.L_main_else_14:
.L_main_endif_15:
    addq $0, %rsp
    callq aether_pgo_reset
    movq $10, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rcx
    addq $16, %rsp
    callq aether_pgo_call_count
    pushq %rax
    movq $0, %rax
    popq %r10
    xchgq %rax, %r10
    cmpq %r10, %rax
    setne %al
    movzbl %al, %eax
    testq %rax, %rax
    je .L_main_else_16
    movq $9, %rax
    addq $80, %rsp
    popq %rbp
    ret
    jmp .L_main_endif_17
.L_main_else_16:
.L_main_endif_17:
    movq $1, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rcx
    addq $16, %rsp
    callq aether_pgo_branch_freq
    movss %xmm0, -48(%rbp)
    movss -48(%rbp), %xmm0
    subq $16, %rsp
    movss %xmm0, (%rsp)
    movss .LF_main_0(%rip), %xmm0
    movss %xmm0, %xmm1
    movss (%rsp), %xmm0
    addq $16, %rsp
    ucomiss %xmm1, %xmm0
    setne %al
    movzbl %al, %eax
    testq %rax, %rax
    je .L_main_else_18
    movq $10, %rax
    addq $80, %rsp
    popq %rbp
    ret
    jmp .L_main_endif_19
.L_main_else_18:
.L_main_endif_19:
    movq $0, %rax
    addq $80, %rsp
    popq %rbp
    ret

