# AETHER x86-64 assembly (Microsoft x64 ABI)
# Emitted by aetherc; comments here are debug-only and do not
# come from any .aether source — those were stripped at lex time.

.section .text
.globl main

main:
    pushq %rbp
    movq %rsp, %rbp
    subq $32, %rsp
    movq $0, %rax
    movq %rax, %rcx
    callq aether_autodiff_init
    callq aether_rt_self_check
    addq $32, %rsp
    popq %rbp
    ret

