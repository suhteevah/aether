# AETHER x86-64 assembly (Microsoft x64 ABI)
# Emitted by aetherc; comments here are debug-only and do not
# come from any .aether source — those were stripped at lex time.

.section .text
.globl main

aether_parse_one:
    pushq %rbp
    movq %rsp, %rbp
    subq $48, %rsp
    movq %rcx, -8(%rbp)
    movq -8(%rbp), %rax
    pushq %rax
    movq $0, %rax
    popq %r10
    xchgq %rax, %r10
    cmpq %r10, %rax
    setg %al
    movzbl %al, %eax
    testq %rax, %rax
    je .L_parse_one_enumret_else_0
    movq -8(%rbp), %rax
    movq %rax, %rdx
    movq $0, %rax
    jmp .L_parse_one_enumret_end_1
.L_parse_one_enumret_else_0:
    movq $99, %rax
    movq %rax, %rdx
    movq $1, %rax
.L_parse_one_enumret_end_1:
    addq $48, %rsp
    popq %rbp
    ret

aether_parse_chain:
    pushq %rbp
    movq %rsp, %rbp
    subq $80, %rsp
    movq %rcx, -8(%rbp)
    movq %rdx, -16(%rbp)
    movq -8(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rcx
    addq $16, %rsp
    callq aether_parse_one
    testq %rax, %rax
    je .L_parse_chain_try_ok_0
    addq $80, %rsp
    popq %rbp
    ret
.L_parse_chain_try_ok_0:
    movq %rdx, %rax
    movq %rax, -32(%rbp)
    movq -16(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rcx
    addq $16, %rsp
    callq aether_parse_one
    testq %rax, %rax
    je .L_parse_chain_try_ok_1
    addq $80, %rsp
    popq %rbp
    ret
.L_parse_chain_try_ok_1:
    movq %rdx, %rax
    movq %rax, -40(%rbp)
    movq -32(%rbp), %rax
    pushq %rax
    movq -40(%rbp), %rax
    popq %r10
    xchgq %rax, %r10
    addq %r10, %rax
    movq %rax, %rdx
    movq $0, %rax
    addq $80, %rsp
    popq %rbp
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $144, %rsp
    movq $20, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $22, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rdx
    movq 16(%rsp), %rax
    movq %rax, %rcx
    addq $32, %rsp
    callq aether_parse_chain
    movq %rax, -16(%rbp)
    movq %rdx, -24(%rbp)
    movq -16(%rbp), %rax
    movq %rax, -32(%rbp)
    movq $0, %r10
    cmpq %r10, %rax
    jne .L_main_match_next_1
    movq -24(%rbp), %rax
    movq %rax, -40(%rbp)
    jmp .L_main_match_end_0
.L_main_match_next_1:
    movq -32(%rbp), %rax
    movq $1, %r10
    cmpq %r10, %rax
    jne .L_main_match_end_0
    movq -24(%rbp), %rax
    movq %rax, -48(%rbp)
    negq %rax
    jmp .L_main_match_end_0
.L_main_match_end_0:
    movq %rax, -56(%rbp)
    movq $1, %rax
    negq %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $5, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rdx
    movq 16(%rsp), %rax
    movq %rax, %rcx
    addq $32, %rsp
    callq aether_parse_chain
    movq %rax, -64(%rbp)
    movq %rdx, -72(%rbp)
    movq -64(%rbp), %rax
    movq %rax, -80(%rbp)
    movq $0, %r10
    cmpq %r10, %rax
    jne .L_main_match_next_3
    movq -72(%rbp), %rax
    movq %rax, -88(%rbp)
    movq $0, %rax
    jmp .L_main_match_end_2
.L_main_match_next_3:
    movq -80(%rbp), %rax
    movq $1, %r10
    cmpq %r10, %rax
    jne .L_main_match_end_2
    movq -72(%rbp), %rax
    movq %rax, -96(%rbp)
    jmp .L_main_match_end_2
.L_main_match_end_2:
    movq %rax, -104(%rbp)
    movq -56(%rbp), %rax
    pushq %rax
    movq -104(%rbp), %rax
    popq %r10
    xchgq %rax, %r10
    subq %r10, %rax
    pushq %rax
    movq $99, %rax
    popq %r10
    xchgq %rax, %r10
    addq %r10, %rax
    addq $144, %rsp
    popq %rbp
    ret

