//! Trait + impl resolution.
//!
//! Phase 6.2 — the compiler today supports `impl Foo { fn ... }` (inherent
//! impls). To reach Rust parity we add explicit `trait` declarations and
//! `impl Trait for Type` blocks. This module is the resolver: it walks
//! collected trait/impl pairs and produces a method-dispatch table keyed
//! by `(type, trait, method)`.
//!
//! Static dispatch (monomorphization) consumes this table directly; trait
//! objects (`dyn Trait`) build a per-(type, trait) vtable from the same
//! table and lay it out as a fat pointer. Phase 6.2 covers the static
//! path only — `dyn` is a separable follow-up.

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct TraitDecl {
    pub name: String,
    pub methods: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ImplBlock {
    pub trait_name: Option<String>,
    pub type_name: String,
    pub method_impls: Vec<(String, String)>,
}

#[derive(Default)]
pub struct Resolver {
    pub traits: HashMap<String, TraitDecl>,
    pub impls: Vec<ImplBlock>,
}

impl Resolver {
    pub fn add_trait(&mut self, t: TraitDecl) { self.traits.insert(t.name.clone(), t); }
    pub fn add_impl(&mut self, i: ImplBlock) { self.impls.push(i); }

    /// Build the per-(type, trait, method) → fn-name dispatch table.
    /// For inherent impls the trait component is None.
    pub fn dispatch_table(&self) -> HashMap<(String, Option<String>, String), String> {
        let mut out = HashMap::new();
        for i in &self.impls {
            for (m, fn_name) in &i.method_impls {
                out.insert(
                    (i.type_name.clone(), i.trait_name.clone(), m.clone()),
                    fn_name.clone(),
                );
            }
        }
        out
    }

    /// Verify every method declared on a trait has a corresponding impl
    /// when the trait appears in `impl Trait for T`.
    pub fn check_completeness(&self) -> Vec<String> {
        let mut errs = Vec::new();
        for i in &self.impls {
            if let Some(tn) = &i.trait_name {
                if let Some(td) = self.traits.get(tn) {
                    for m in &td.methods {
                        if !i.method_impls.iter().any(|(mm, _)| mm == m) {
                            errs.push(format!(
                                "impl {} for {}: missing method `{}`",
                                tn, i.type_name, m));
                        }
                    }
                } else {
                    errs.push(format!("impl {}: unknown trait `{}`", i.type_name, tn));
                }
            }
        }
        errs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(x: &str) -> String { x.into() }

    #[test]
    fn dispatch_table_keys_by_type_trait_method() {
        let mut r = Resolver::default();
        r.add_trait(TraitDecl { name: s("Add"), methods: vec![s("add")] });
        r.add_impl(ImplBlock {
            trait_name: Some(s("Add")),
            type_name: s("Vec3"),
            method_impls: vec![(s("add"), s("Vec3__Add__add"))],
        });
        let t = r.dispatch_table();
        assert_eq!(t[&(s("Vec3"), Some(s("Add")), s("add"))], "Vec3__Add__add");
    }

    #[test]
    fn missing_method_flagged() {
        let mut r = Resolver::default();
        r.add_trait(TraitDecl { name: s("Show"), methods: vec![s("show"), s("debug")] });
        r.add_impl(ImplBlock {
            trait_name: Some(s("Show")),
            type_name: s("Foo"),
            method_impls: vec![(s("show"), s("Foo__Show__show"))],
        });
        let errs = r.check_completeness();
        assert_eq!(errs.len(), 1);
        assert!(errs[0].contains("debug"));
    }

    #[test]
    fn unknown_trait_flagged() {
        let mut r = Resolver::default();
        r.add_impl(ImplBlock {
            trait_name: Some(s("Nope")),
            type_name: s("Foo"),
            method_impls: vec![],
        });
        let errs = r.check_completeness();
        assert_eq!(errs.len(), 1);
        assert!(errs[0].contains("unknown trait"));
    }
}
