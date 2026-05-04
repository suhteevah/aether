# AETHER x86-64 assembly (Microsoft x64 ABI)
# Emitted by aetherc; comments here are debug-only and do not
# come from any .aether source — those were stripped at lex time.

.section .text
.globl main

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $112, %rsp
    addq $0, %rsp
    callq aether_vec_i64_new
    movq %rax, -16(%rbp)
    pushq %rax
    movq $0, %rax
    popq %r10
    xchgq %rax, %r10
    cmpq %r10, %rax
    setl %al
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
    movq $0, -24(%rbp)
.L_main_while_top_2:
    movq -24(%rbp), %rax
    pushq %rax
    movq $10, %rax
    popq %r10
    xchgq %rax, %r10
    cmpq %r10, %rax
    setl %al
    movzbl %al, %eax
    testq %rax, %rax
    je .L_main_while_end_3
    movq -16(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq -24(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rdx
    movq 16(%rsp), %rax
    movq %rax, %rcx
    addq $32, %rsp
    callq aether_vec_i64_push
    pushq %rax
    movq $0, %rax
    popq %r10
    xchgq %rax, %r10
    cmpq %r10, %rax
    setne %al
    movzbl %al, %eax
    testq %rax, %rax
    je .L_main_else_4
    movq $2, %rax
    addq $64, %rsp
    popq %rbp
    ret
    jmp .L_main_endif_5
.L_main_else_4:
.L_main_endif_5:
    movq -24(%rbp), %rax
    pushq %rax
    movq $1, %rax
    popq %r10
    xchgq %rax, %r10
    addq %r10, %rax
    movq %rax, -24(%rbp)
    jmp .L_main_while_top_2
.L_main_while_end_3:
    xorl %eax, %eax
    movq -16(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rcx
    addq $16, %rsp
    callq aether_vec_i64_len
    pushq %rax
    movq $10, %rax
    popq %r10
    xchgq %rax, %r10
    cmpq %r10, %rax
    setne %al
    movzbl %al, %eax
    testq %rax, %rax
    je .L_main_else_6
    movq $3, %rax
    addq $64, %rsp
    popq %rbp
    ret
    jmp .L_main_endif_7
.L_main_else_6:
.L_main_endif_7:
    movq $0, -32(%rbp)
    movq $0, -40(%rbp)
.L_main_while_top_8:
    movq -40(%rbp), %rax
    pushq %rax
    movq $10, %rax
    popq %r10
    xchgq %rax, %r10
    cmpq %r10, %rax
    setl %al
    movzbl %al, %eax
    testq %rax, %rax
    je .L_main_while_end_9
    movq -32(%rbp), %rax
    pushq %rax
    movq -16(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq -40(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rdx
    movq 16(%rsp), %rax
    movq %rax, %rcx
    addq $32, %rsp
    callq aether_vec_i64_get
    popq %r10
    xchgq %rax, %r10
    addq %r10, %rax
    movq %rax, -32(%rbp)
    movq -40(%rbp), %rax
    pushq %rax
    movq $1, %rax
    popq %r10
    xchgq %rax, %r10
    addq %r10, %rax
    movq %rax, -40(%rbp)
    jmp .L_main_while_top_8
.L_main_while_end_9:
    xorl %eax, %eax
    movq -32(%rbp), %rax
    pushq %rax
    movq $45, %rax
    popq %r10
    xchgq %rax, %r10
    cmpq %r10, %rax
    setne %al
    movzbl %al, %eax
    testq %rax, %rax
    je .L_main_else_10
    movq $4, %rax
    addq $80, %rsp
    popq %rbp
    ret
    jmp .L_main_endif_11
.L_main_else_10:
.L_main_endif_11:
    movq -16(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $3, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $999, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %r8
    movq 16(%rsp), %rax
    movq %rax, %rdx
    movq 32(%rsp), %rax
    movq %rax, %rcx
    addq $48, %rsp
    callq aether_vec_i64_set
    pushq %rax
    movq $0, %rax
    popq %r10
    xchgq %rax, %r10
    cmpq %r10, %rax
    setne %al
    movzbl %al, %eax
    testq %rax, %rax
    je .L_main_else_12
    movq $5, %rax
    addq $80, %rsp
    popq %rbp
    ret
    jmp .L_main_endif_13
.L_main_else_12:
.L_main_endif_13:
    movq -16(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $3, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rdx
    movq 16(%rsp), %rax
    movq %rax, %rcx
    addq $32, %rsp
    callq aether_vec_i64_get
    pushq %rax
    movq $999, %rax
    popq %r10
    xchgq %rax, %r10
    cmpq %r10, %rax
    setne %al
    movzbl %al, %eax
    testq %rax, %rax
    je .L_main_else_14
    movq $6, %rax
    addq $80, %rsp
    popq %rbp
    ret
    jmp .L_main_endif_15
.L_main_else_14:
.L_main_endif_15:
    movq -16(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rcx
    addq $16, %rsp
    callq aether_vec_i64_free
    pushq %rax
    movq $0, %rax
    popq %r10
    xchgq %rax, %r10
    cmpq %r10, %rax
    setne %al
    movzbl %al, %eax
    testq %rax, %rax
    je .L_main_else_16
    movq $7, %rax
    addq $80, %rsp
    popq %rbp
    ret
    jmp .L_main_endif_17
.L_main_else_16:
.L_main_endif_17:
    addq $0, %rsp
    callq aether_string_new
    movq %rax, -48(%rbp)
    pushq %rax
    movq $0, %rax
    popq %r10
    xchgq %rax, %r10
    cmpq %r10, %rax
    setl %al
    movzbl %al, %eax
    testq %rax, %rax
    je .L_main_else_18
    movq $8, %rax
    addq $80, %rsp
    popq %rbp
    ret
    jmp .L_main_endif_19
.L_main_else_18:
.L_main_endif_19:
    movq -48(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $72, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rdx
    movq 16(%rsp), %rax
    movq %rax, %rcx
    addq $32, %rsp
    callq aether_string_push_byte
    pushq %rax
    movq $0, %rax
    popq %r10
    xchgq %rax, %r10
    cmpq %r10, %rax
    setne %al
    movzbl %al, %eax
    testq %rax, %rax
    je .L_main_else_20
    movq $9, %rax
    addq $80, %rsp
    popq %rbp
    ret
    jmp .L_main_endif_21
.L_main_else_20:
.L_main_endif_21:
    movq -48(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $105, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rdx
    movq 16(%rsp), %rax
    movq %rax, %rcx
    addq $32, %rsp
    callq aether_string_push_byte
    pushq %rax
    movq $0, %rax
    popq %r10
    xchgq %rax, %r10
    cmpq %r10, %rax
    setne %al
    movzbl %al, %eax
    testq %rax, %rax
    je .L_main_else_22
    movq $10, %rax
    addq $80, %rsp
    popq %rbp
    ret
    jmp .L_main_endif_23
.L_main_else_22:
.L_main_endif_23:
    movq -48(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rcx
    addq $16, %rsp
    callq aether_string_len
    pushq %rax
    movq $2, %rax
    popq %r10
    xchgq %rax, %r10
    cmpq %r10, %rax
    setne %al
    movzbl %al, %eax
    testq %rax, %rax
    je .L_main_else_24
    movq $11, %rax
    addq $80, %rsp
    popq %rbp
    ret
    jmp .L_main_endif_25
.L_main_else_24:
.L_main_endif_25:
    movq -48(%rbp), %rax
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
    callq aether_string_byte_at
    movq %rax, -56(%rbp)
    movq -48(%rbp), %rax
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
    callq aether_string_byte_at
    movq %rax, -64(%rbp)
    movq -56(%rbp), %rax
    pushq %rax
    movq $72, %rax
    popq %r10
    xchgq %rax, %r10
    cmpq %r10, %rax
    setne %al
    movzbl %al, %eax
    testq %rax, %rax
    je .L_main_else_26
    movq $12, %rax
    addq $96, %rsp
    popq %rbp
    ret
    jmp .L_main_endif_27
.L_main_else_26:
.L_main_endif_27:
    movq -64(%rbp), %rax
    pushq %rax
    movq $105, %rax
    popq %r10
    xchgq %rax, %r10
    cmpq %r10, %rax
    setne %al
    movzbl %al, %eax
    testq %rax, %rax
    je .L_main_else_28
    movq $13, %rax
    addq $96, %rsp
    popq %rbp
    ret
    jmp .L_main_endif_29
.L_main_else_28:
.L_main_endif_29:
    movq -48(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rcx
    addq $16, %rsp
    callq aether_string_free
    pushq %rax
    movq $0, %rax
    popq %r10
    xchgq %rax, %r10
    cmpq %r10, %rax
    setne %al
    movzbl %al, %eax
    testq %rax, %rax
    je .L_main_else_30
    movq $14, %rax
    addq $96, %rsp
    popq %rbp
    ret
    jmp .L_main_endif_31
.L_main_else_30:
.L_main_endif_31:
    movq -32(%rbp), %rax
    pushq %rax
    movq -56(%rbp), %rax
    popq %r10
    xchgq %rax, %r10
    addq %r10, %rax
    pushq %rax
    movq -64(%rbp), %rax
    popq %r10
    xchgq %rax, %r10
    addq %r10, %rax
    pushq %rax
    movq $180, %rax
    popq %r10
    xchgq %rax, %r10
    subq %r10, %rax
    movq %rax, -72(%rbp)
    addq $112, %rsp
    popq %rbp
    ret
    xorl %eax, %eax
    addq $112, %rsp
    popq %rbp
    ret

