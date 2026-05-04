# AETHER x86-64 assembly (Microsoft x64 ABI)
# Emitted by aetherc; comments here are debug-only and do not
# come from any .aether source — those were stripped at lex time.

.section .text
.globl main

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $128, %rsp
    movq $2, -16(%rbp)
    movq $4, -24(%rbp)
    movq $6, -32(%rbp)
    movq $12, -40(%rbp)
    movq $18, -48(%rbp)
    movq -16(%rbp), %rax
    movq %rax, -56(%rbp)
    movq -24(%rbp), %rax
    movq %rax, -64(%rbp)
    movq -32(%rbp), %rax
    movq %rax, -72(%rbp)
    movq -40(%rbp), %rax
    movq %rax, -80(%rbp)
    movq -48(%rbp), %rax
    movq %rax, -88(%rbp)
    movq -56(%rbp), %rax
    pushq %rax
    movq -64(%rbp), %rax
    popq %r10
    xchgq %rax, %r10
    addq %r10, %rax
    pushq %rax
    movq -72(%rbp), %rax
    popq %r10
    xchgq %rax, %r10
    addq %r10, %rax
    pushq %rax
    movq -80(%rbp), %rax
    popq %r10
    xchgq %rax, %r10
    addq %r10, %rax
    pushq %rax
    movq -88(%rbp), %rax
    popq %r10
    xchgq %rax, %r10
    addq %r10, %rax
    addq $128, %rsp
    popq %rbp
    ret

