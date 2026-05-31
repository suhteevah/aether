//! Phase 6.2 — drive `mir::traits::Resolver` over the program.
//!
//! The asm backend's name-mangling flattener turns `impl Trait for Type {
//! fn m(&self) … }` into a top-level `Type__m`, which is enough for method
//! dispatch (`obj.m()` → `Type__m(obj)`). But two real trait semantics need
//! the resolver, and neither falls out of mangling:
//!
//! 1. **Default methods.** A `trait` method declared WITH a body is a
//!    default. An `impl Trait for Type` that omits it inherits the default —
//!    we clone the default `FnDecl` into the impl so the flattener emits
//!    `Type__m`. Before this pass the default body was dropped entirely
//!    (`Item::Trait{..} => {}` in the asm backend), so an impl that relied on
//!    a default produced an undefined-symbol call.
//!
//! 2. **Completeness checking.** An impl that omits a *required* (bodyless)
//!    trait method, or impls a trait that was never declared, is a hard
//!    error — `AE0210` / `AE0211` — instead of silently compiling a type
//!    that's missing part of its interface.
//!
//! The required-method completeness check is delegated to
//! `Resolver::check_completeness`, so the resolver is now genuinely on the
//! compile path rather than an untested island.

use crate::ast::{FnDecl, Item, Program};
use crate::diag::Diag;
use super::traits::{ImplBlock, Resolver, TraitDecl};
use std::collections::{HashMap, HashSet};

pub struct TraitReport {
    /// Number of default-method `FnDecl`s spliced into impls.
    pub synthesized_defaults: usize,
    /// Completeness / unknown-trait diagnostics (AE0210 / AE0211).
    pub diags: Vec<Diag>,
}

pub fn run(prog: &mut Program) -> TraitReport {
    // 1. Collect trait declarations (full FnDecls, so default bodies survive)
    //    + each trait's supertrait bounds.
    let mut trait_methods: HashMap<String, Vec<FnDecl>> = HashMap::new();
    let mut supertraits: HashMap<String, Vec<String>> = HashMap::new();
    for it in &prog.items {
        if let Item::Trait { name, methods, supertraits: st } = it {
            trait_methods.insert(name.clone(), methods.clone());
            supertraits.insert(name.clone(), st.clone());
        }
    }
    // Which (type) implements which trait — for supertrait satisfaction checks.
    let mut impls_of: HashMap<String, HashSet<String>> = HashMap::new();
    for it in &prog.items {
        if let Item::ImplTrait { trait_name, type_name, .. } = it {
            impls_of.entry(type_name.clone()).or_default().insert(trait_name.clone());
        }
    }

    // 2. Synthesize default-method impls into each `impl Trait for Type` that
    //    omits a method the trait supplies a default body for.
    let mut synthesized_defaults = 0usize;
    for it in prog.items.iter_mut() {
        if let Item::ImplTrait { trait_name, methods, .. } = it {
            if let Some(tms) = trait_methods.get(trait_name) {
                let provided: HashSet<String> =
                    methods.iter().map(|m| m.name.clone()).collect();
                for tm in tms {
                    if tm.body.is_some() && !provided.contains(&tm.name) {
                        methods.push(tm.clone());
                        synthesized_defaults += 1;
                    }
                }
            }
        }
    }

    // 3. Build the resolver and run completeness against the REQUIRED
    //    (bodyless) trait methods. After step 2 the impl's `methods` already
    //    carries any synthesized defaults, so only a truly-omitted required
    //    method (or an unknown trait) remains for the resolver to flag.
    let mut r = Resolver::default();
    for (name, methods) in &trait_methods {
        let required: Vec<String> = methods.iter()
            .filter(|m| m.body.is_none())
            .map(|m| m.name.clone())
            .collect();
        r.add_trait(TraitDecl { name: name.clone(), methods: required });
    }
    for it in &prog.items {
        if let Item::ImplTrait { trait_name, type_name, methods } = it {
            // Marker-trait opt-in: an EMPTY impl of an undeclared trait
            // (`unsafe impl Send for Foo {}`) has no interface to verify —
            // allow it silently. A *non-empty* impl of an undeclared trait is
            // a real unknown-trait error (typically a typo), so it still goes
            // to the resolver and surfaces as AE0211.
            let declared = trait_methods.contains_key(trait_name);
            if !declared && methods.is_empty() {
                continue;
            }
            r.add_impl(ImplBlock {
                trait_name: Some(trait_name.clone()),
                type_name: type_name.clone(),
                method_impls: methods.iter()
                    .map(|m| (m.name.clone(), format!("{}__{}", type_name, m.name)))
                    .collect(),
            });
        }
    }

    let mut diags = Vec::new();
    for msg in r.check_completeness() {
        let (code, hint): (&'static str, &'static str) = if msg.contains("unknown trait") {
            ("AE0211",
             "declare the trait with `trait <Name> { ... }`, or fix the trait name in the impl")
        } else {
            ("AE0210",
             "the trait requires this method; add it to the impl block, or give the \
              trait method a default body so impls can inherit it")
        };
        diags.push(Diag::error(code, "trait", msg).with_hint(hint));
    }

    // Supertrait satisfaction: `impl Pet for Dog` requires `impl Animal for Dog`
    // when `trait Pet: Animal`. Missing supertrait impl -> AE0212.
    for it in &prog.items {
        if let Item::ImplTrait { trait_name, type_name, .. } = it {
            let Some(supers) = supertraits.get(trait_name) else { continue; };
            let provided = impls_of.get(type_name);
            for sup in supers {
                let has = provided.map_or(false, |s| s.contains(sup));
                if !has {
                    diags.push(Diag::error("AE0212", "trait",
                        format!("`{}` requires supertrait `{}`, but `{}` does not implement it",
                            trait_name, sup, type_name))
                        .with_hint(format!("add `impl {} for {} {{ ... }}` so the supertrait \
                            bound on `{}` is satisfied", sup, type_name, trait_name)));
                }
            }
        }
    }

    TraitReport { synthesized_defaults, diags }
}
