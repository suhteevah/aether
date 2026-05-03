# AETHER x86-64 assembly (Microsoft x64 ABI)
# Emitted by aetherc; comments here are debug-only and do not
# come from any .aether source — those were stripped at lex time.

.section .text
.globl main

aether_Foo__sum_plus:
    pushq %rbp
    movq %rsp, %rbp
    subq $64, %rsp
    movq %rcx, -8(%rbp)
    movq %rdx, -16(%rbp)
    movq %r8, -24(%rbp)
    movq -8(%rbp), %rax
    pushq %rax
    movq -16(%rbp), %rax
    popq %r10
    xchgq %rax, %r10
    addq %r10, %rax
    pushq %rax
    movq -24(%rbp), %rax
    popq %r10
    xchgq %rax, %r10
    addq %r10, %rax
    addq $64, %rsp
    popq %rbp
    ret

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $64, %rsp
    movq $10, %rax
    movq %rax, -16(%rbp)
    movq $20, %rax
    movq %rax, -24(%rbp)
    movq -16(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq -24(%rbp), %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq $12, %rax
    subq $16, %rsp
    movq %rax, (%rsp)
    movq 0(%rsp), %rax
    movq %rax, %rdx
    movq 16(%rsp), %rax
    movq %rax, %rcx
    addq $32, %rsp
    callq aether_Foo__sum_plus
    addq $64, %rsp
    popq %rbp
    ret

