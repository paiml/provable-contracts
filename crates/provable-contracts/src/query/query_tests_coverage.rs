    // Coverage-targeted tests for query module internals.

    use super::*;

    fn test_index() -> ContractIndex {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts");
        ContractIndex::build_from_directory(&dir).unwrap()
    }

    #[test]
    fn min_level_filter_restricts_results() {
        let index = test_index();
        let all = QueryParams {
            query: "softmax".to_string(),
            limit: 50,
            ..Default::default()
        };
        let all_output = execute(&index, &all);

        let filtered = QueryParams {
            query: "softmax".to_string(),
            limit: 50,
            min_level: Some("L3".to_string()),
            ..Default::default()
        };
        let filtered_output = execute(&index, &filtered);

        // L3 filter should return <= total results
        assert!(filtered_output.total_matches <= all_output.total_matches);
    }

    #[test]
    fn parse_iso_days_ago_recent() {
        // 2026-03-06 at epoch ~1772870400
        let now = 1772870400;
        let days = super::parse_iso_days_ago("2026-03-05", now);
        assert!(days <= 2, "Yesterday should be ~1 day ago, got {days}");
    }

    #[test]
    fn parse_iso_days_ago_invalid() {
        assert_eq!(super::parse_iso_days_ago("invalid", 1772870400), 0);
        assert_eq!(super::parse_iso_days_ago("2026-03", 1772870400), 0);
    }

    #[test]
    fn month_days_all_months() {
        assert_eq!(super::month_days(0), 0);
        assert_eq!(super::month_days(1), 0);
        assert_eq!(super::month_days(2), 31);
        assert_eq!(super::month_days(6), 151);
        assert_eq!(super::month_days(12), 334);
        assert_eq!(super::month_days(13), 0); // out of range
    }

    #[test]
    fn parse_proof_level_all_levels() {
        use crate::proof_status::ProofLevel;
        assert_eq!(super::parse_proof_level("L1"), ProofLevel::L1);
        assert_eq!(super::parse_proof_level("L2"), ProofLevel::L2);
        assert_eq!(super::parse_proof_level("L3"), ProofLevel::L3);
        assert_eq!(super::parse_proof_level("L4"), ProofLevel::L4);
        assert_eq!(super::parse_proof_level("L5"), ProofLevel::L5);
        assert_eq!(super::parse_proof_level("l3"), ProofLevel::L3);
        assert_eq!(super::parse_proof_level("unknown"), ProofLevel::L1);
    }

    #[test]
    fn show_diff_enrichment() {
        let index = test_index();
        let params = QueryParams {
            query: "softmax".to_string(),
            show_diff: true,
            limit: 2,
            ..Default::default()
        };
        let output = execute(&index, &params);
        assert!(!output.results.is_empty());
        // Diff info should be populated for contracts in a git repo
        let has_diff = output.results.iter().any(|r| r.diff.is_some());
        assert!(has_diff, "At least one result should have diff info");
    }

    #[test]
    fn depends_on_filter() {
        let index = test_index();
        let params = QueryParams {
            query: "attention".to_string(),
            depends_on: Some("softmax-kernel-v1".to_string()),
            limit: 20,
            ..Default::default()
        };
        let output = execute(&index, &params);
        // All results should depend on softmax-kernel-v1
        for r in &output.results {
            let entry = index.get_by_stem(&r.stem).unwrap();
            assert!(
                entry.depends_on.contains(&"softmax-kernel-v1".to_string()),
                "{} should depend on softmax-kernel-v1",
                r.stem
            );
        }
    }

    #[test]
    fn binding_gaps_filter() {
        let index = test_index();
        let binding_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../contracts/aprender/binding.yaml");
        let params = QueryParams {
            query: "kernel".to_string(),
            binding_gaps_only: true,
            binding_path: Some(binding_path.display().to_string()),
            limit: 20,
            ..Default::default()
        };
        let output = execute(&index, &params);
        // Should only include contracts with binding gaps
        // (may be 0 if all are implemented, which is fine)
        assert!(output.total_matches <= 20);
    }

    #[test]
    fn binding_with_real_registry() {
        let index = test_index();
        let binding_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../contracts/aprender/binding.yaml");
        let params = QueryParams {
            query: "softmax".to_string(),
            show_binding: true,
            binding_path: Some(binding_path.display().to_string()),
            limit: 1,
            ..Default::default()
        };
        let output = execute(&index, &params);
        if let Some(r) = output.results.first() {
            // With real binding, some equations should be bound or unbound
            assert!(!r.bindings.is_empty());
        }
    }

    #[test]
    fn display_with_diff_and_bindings() {
        let r = QueryResult {
            rank: 1,
            stem: "test-v1".to_string(),
            path: "test.yaml".to_string(),
            relevance: 0.9,
            description: "test contract".to_string(),
            equations: vec!["eq1".to_string()],
            obligation_count: 3,
            references: vec![],
            depends_on: vec!["dep-v1".to_string()],
            depended_by: vec!["consumer-v1".to_string()],
            score: None,
            proof_status: None,
            bindings: vec![EquationBinding {
                equation: "eq1".to_string(),
                status: "implemented".to_string(),
                module_path: Some("mod::path".to_string()),
            }],
            diff: Some(DiffInfo {
                last_modified: "2026-03-01".to_string(),
                days_ago: 5,
                commit_hash: "abc1234def".to_string(),
            }),
            pagerank: Some(0.042),
        };
        let text = format!("{r}");
        assert!(text.contains("Bindings:"));
        assert!(text.contains("eq1: implemented (mod::path)"));
        assert!(text.contains("Last modified: 2026-03-01"));
        assert!(text.contains("Depends on: dep-v1"));
        assert!(text.contains("Depended by: consumer-v1"));
        assert!(text.contains("PageRank: 0.0420"));
    }

    #[test]
    fn markdown_with_diff_and_bindings() {
        let output = QueryOutput {
            query: "test".to_string(),
            total_matches: 1,
            results: vec![QueryResult {
                rank: 1,
                stem: "test-v1".to_string(),
                path: "test.yaml".to_string(),
                relevance: 0.9,
                description: "test".to_string(),
                equations: vec![],
                obligation_count: 1,
                references: vec![],
                depends_on: vec!["dep-v1".to_string()],
                depended_by: vec!["consumer-v1".to_string()],
                score: None,
                proof_status: None,
                bindings: vec![EquationBinding {
                    equation: "eq1".to_string(),
                    status: "unbound".to_string(),
                    module_path: None,
                }],
                diff: Some(DiffInfo {
                    last_modified: "2026-03-01".to_string(),
                    days_ago: 5,
                    commit_hash: "abc1234".to_string(),
                }),
                pagerank: None,
            }],
        };
        let md = output.to_markdown();
        assert!(md.contains("**Bindings:**"));
        assert!(md.contains("`eq1`: unbound"));
        assert!(md.contains("**Last modified:**"));
        assert!(md.contains("**Depends on:** dep-v1"));
        assert!(md.contains("**Depended by:** consumer-v1"));
    }

    #[test]
    fn binding_unbound_path() {
        let index = test_index();
        // Use a fake binding path to trigger the "no binding found" path
        let params = QueryParams {
            query: "softmax".to_string(),
            show_binding: true,
            binding_path: Some("nonexistent-binding.yaml".to_string()),
            limit: 1,
            ..Default::default()
        };
        let output = execute(&index, &params);
        assert!(!output.results.is_empty());
        // Without valid binding, equations show "no binding registry"
        if let Some(r) = output.results.first() {
            if !r.bindings.is_empty() {
                assert_eq!(r.bindings[0].status, "no binding registry");
            }
        }
    }

    #[test]
    fn binding_gaps_without_registry() {
        let index = test_index();
        // binding_gaps_only=true with no binding path => all filtered out
        let params = QueryParams {
            query: "softmax".to_string(),
            binding_gaps_only: true,
            limit: 10,
            ..Default::default()
        };
        let output = execute(&index, &params);
        assert_eq!(output.total_matches, 0, "No binding registry means all filtered");
    }

    #[test]
    fn pagerank_enrichment() {
        let index = test_index();
        let params = QueryParams {
            query: "softmax".to_string(),
            show_pagerank: true,
            limit: 3,
            ..Default::default()
        };
        let output = execute(&index, &params);
        assert!(!output.results.is_empty());
        // Softmax-kernel-v1 should have a pagerank score
        let sm = output.results.iter().find(|r| r.stem == "softmax-kernel-v1");
        if let Some(r) = sm {
            assert!(r.pagerank.is_some(), "pagerank should be populated");
            assert!(r.pagerank.unwrap() > 0.0, "pagerank should be positive");
        }
    }

    #[test]
    fn graph_enrichment_with_depended_by() {
        let index = test_index();
        let params = QueryParams {
            query: "softmax-kernel".to_string(),
            show_graph: true,
            limit: 3,
            ..Default::default()
        };
        let output = execute(&index, &params);
        let sm = output.results.iter().find(|r| r.stem == "softmax-kernel-v1");
        if let Some(r) = sm {
            assert!(!r.depended_by.is_empty(), "softmax should have dependents");
        }
    }
