# AETHER x86-64 assembly (Microsoft x64 ABI)
# Emitted by aetherc; comments here are debug-only and do not
# come from any .aether source — those were stripped at lex time.

.section .text
.globl main

aether_hot_path:
    pushq %rbp
    movq %rsp, %rbp
    subq $48, %rsp
    movq $42, %rax
    addq $48, %rsp
    popq %rbp
    ret

.section .text.cold,"x"
aether_rare_path:
    pushq %rbp
    movq %rsp, %rbp
    subq $48, %rsp
    movq $99, %rax
    addq $48, %rsp
    popq %rbp
    ret

.section .text,"x"
main:
    pushq %rbp
    movq %rsp, %rbp
    subq $64, %rsp
    addq $0, %rsp
    callq aether_hot_path
    movq %rax, -16(%rbp)
    pushq %rax
    movq $0, %rax
    popq %r10
    xchgq %rax, %r10
    cmpq %r10, %rax
    sete %al
    movzbl %al, %eax
    testq %rax, %rax
    je .L_main_else_0
    addq $0, %rsp
    callq aether_rare_path
    movq %rax, -24(%rbp)
    jmp .L_main_endif_1
.L_main_else_0:
.L_main_endif_1:
    movq -16(%rbp), %rax
    addq $64, %rsp
    popq %rbp
    ret

