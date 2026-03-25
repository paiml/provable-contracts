use super::*;

#[test]
fn test_extract_bound_functions() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("binding.yaml");
    std::fs::write(
        &path,
        "bindings:\n  - function: \"Foo::bar\"\n    status: implemented\n  - function: baz\n    status: implemented\n",
    )
    .unwrap();
    let names = extract_bound_functions(&path);
    assert!(names.contains("bar"), "Expected 'bar' in {names:?}");
    assert!(names.contains("baz"), "Expected 'baz' in {names:?}");
}

#[test]
fn test_scan_file() {
    let tmp = tempfile::NamedTempFile::with_suffix(".rs").unwrap();
    std::fs::write(
        tmp.path(),
        "pub fn hello() {}\n#[contract(\"test\", equation = \"eq\")]\npub fn world() {}\nfn private() {}\n",
    )
    .unwrap();
    let mut results = Vec::new();
    scan_file(tmp.path(), &mut results);
    assert_eq!(results.len(), 2);
    assert!(!results[0].has_contract_macro);
    assert!(results[1].has_contract_macro);
}
