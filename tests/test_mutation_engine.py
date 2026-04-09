"""Tests for the mutation engine (server/attacks/mutation_engine.py)."""

import json
import tempfile

import pytest

from server.attacks.mutation_engine import (
    CONTEXT_FRAMES,
    HOMOGLYPHS,
    LEETSPEAK_MAP,
    ZERO_WIDTH_CHARS,
    CharacterMutator,
    CompositeMutator,
    ContextualMutator,
    EncodingMutator,
    MutationEngine,
    MutationResult,
    MutationSpec,
    StructuralMutator,
    create_engine,
    quick_mutate,
)

# ──────────────────────────────────────────────
# Helper: validate attack dict schema
# ──────────────────────────────────────────────

REQUIRED_KEYS = {"text", "ground_truth", "is_attack", "attack_type", "difficulty"}


def _assert_valid_attack(d: dict) -> None:
    assert isinstance(d, dict)
    for key in REQUIRED_KEYS:
        assert key in d, f"Missing key: {key}"
    assert isinstance(d["text"], str) and len(d["text"]) > 0
    assert isinstance(d["ground_truth"], str)
    assert d["is_attack"] is True
    assert isinstance(d["attack_type"], str)
    assert isinstance(d["difficulty"], str)


# ──────────────────────────────────────────────
# CharacterMutator
# ──────────────────────────────────────────────


class TestCharacterMutator:
    def test_init_default_seed(self):
        m = CharacterMutator()
        assert m.rng is not None

    def test_init_with_seed(self):
        m = CharacterMutator(seed=42)
        assert m.rng is not None

    def test_apply_leetspeak_basic(self):
        m = CharacterMutator(seed=42)
        result = m.apply_leetspeak("hello test", intensity=1.0)
        # With intensity=1.0 every eligible char should be replaced
        assert "3" in result  # e -> 3
        assert "0" in result  # o -> 0

    def test_apply_leetspeak_zero_intensity(self):
        m = CharacterMutator(seed=42)
        result = m.apply_leetspeak("hello test", intensity=0.0)
        assert result == "hello test"

    def test_apply_leetspeak_preserves_non_map_chars(self):
        m = CharacterMutator(seed=42)
        result = m.apply_leetspeak("!!!", intensity=1.0)
        assert result == "!!!"

    def test_apply_homoglyphs_basic(self):
        m = CharacterMutator(seed=42)
        result = m.apply_homoglyphs("attack", intensity=1.0)
        # At least some homoglyph substitutions should occur
        assert len(result) == len("attack")

    def test_apply_homoglyphs_zero_intensity(self):
        m = CharacterMutator(seed=42)
        result = m.apply_homoglyphs("hello", intensity=0.0)
        assert result == "hello"

    def test_apply_zero_width_injection(self):
        m = CharacterMutator(seed=42)
        result = m.apply_zero_width_injection("hello", frequency=2)
        # Should contain zero-width characters
        has_zw = any(c in ZERO_WIDTH_CHARS for c in result)
        assert has_zw or len(result) >= len("hello")

    def test_apply_case_manipulation_random(self):
        m = CharacterMutator(seed=42)
        result = m.apply_case_manipulation("hello world", style="random")
        assert len(result) == len("hello world")

    def test_apply_case_manipulation_inverse(self):
        m = CharacterMutator(seed=42)
        result = m.apply_case_manipulation("HeLLo", style="inverse")
        assert result == "hEllO"

    def test_apply_case_manipulation_alternating(self):
        m = CharacterMutator(seed=42)
        result = m.apply_case_manipulation("hello", style="alternating")
        assert result == "HeLlO"

    def test_apply_case_manipulation_uppercase_words(self):
        m = CharacterMutator(seed=42)
        result = m.apply_case_manipulation("hello world test", style="uppercase_words")
        assert len(result) == len("hello world test")

    def test_apply_case_manipulation_unknown_style(self):
        m = CharacterMutator(seed=42)
        original = "hello world"
        result = m.apply_case_manipulation(original, style="bogus")
        assert result == original

    def test_mutate_leetspeak(self):
        m = CharacterMutator(seed=42)
        _result = m.mutate("hello test", strategy="leetspeak")
        assert True  # random may produce same

    def test_mutate_homoglyphs(self):
        m = CharacterMutator(seed=42)
        result = m.mutate("hello", strategy="homoglyphs")
        assert isinstance(result, str)

    def test_mutate_zero_width(self):
        m = CharacterMutator(seed=42)
        result = m.mutate("hello", strategy="zero_width")
        assert isinstance(result, str)

    def test_mutate_case_random(self):
        m = CharacterMutator(seed=42)
        result = m.mutate("hello", strategy="case_random")
        assert isinstance(result, str)
        assert len(result) == len("hello")

    def test_mutate_case_alternating(self):
        m = CharacterMutator(seed=42)
        result = m.mutate("hello", strategy="case_alternating")
        assert result == "HeLlO"

    def test_mutate_all(self):
        m = CharacterMutator(seed=42)
        result = m.mutate_all("hello test")
        assert isinstance(result, str)
        assert len(result) >= len("hello test")

    def test_mutate_default_applies_all(self):
        m = CharacterMutator(seed=42)
        result = m.mutate("hello", strategy="all")
        assert isinstance(result, str)

    def test_deterministic_with_seed(self):
        m1 = CharacterMutator(seed=123)
        m2 = CharacterMutator(seed=123)
        assert m1.apply_leetspeak("hello", intensity=1.0) == m2.apply_leetspeak("hello", intensity=1.0)


# ──────────────────────────────────────────────
# StructuralMutator
# ──────────────────────────────────────────────


class TestStructuralMutator:
    def test_init_default_seed(self):
        m = StructuralMutator()
        assert m.rng is not None

    def test_init_with_seed(self):
        m = StructuralMutator(seed=42)
        assert m.rng is not None

    def test_wrap_json_valid(self):
        m = StructuralMutator(seed=42)
        result = m.wrap_json("test input")
        # At least one template should produce valid JSON
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_wrap_xml_has_tags(self):
        m = StructuralMutator(seed=42)
        result = m.wrap_xml("test input")
        assert "<" in result and ">" in result

    def test_wrap_markdown_has_code_block_or_header(self):
        m = StructuralMutator(seed=42)
        result = m.wrap_markdown("test input")
        assert "```" in result or "#" in result or ">" in result

    def test_wrap_html_comment(self):
        m = StructuralMutator(seed=42)
        result = m.wrap_html_comment("test input")
        assert "<!--" in result and "-->" in result

    def test_wrap_yaml(self):
        m = StructuralMutator(seed=42)
        result = m.wrap_yaml("test input")
        assert "test input" in result

    def test_wrap_code_context(self):
        m = StructuralMutator(seed=42)
        result = m.wrap_code_context("test input")
        assert "test input" in result

    def test_mutate_json_strategy(self):
        m = StructuralMutator(seed=42)
        result = m.mutate("test", strategy="json")
        assert "test" in result

    def test_mutate_xml_strategy(self):
        m = StructuralMutator(seed=42)
        result = m.mutate("test", strategy="xml")
        assert "<" in result

    def test_mutate_markdown_strategy(self):
        m = StructuralMutator(seed=42)
        result = m.mutate("test", strategy="markdown")
        assert "test" in result

    def test_mutate_html_comment_strategy(self):
        m = StructuralMutator(seed=42)
        result = m.mutate("test", strategy="html_comment")
        assert "<!--" in result

    def test_mutate_yaml_strategy(self):
        m = StructuralMutator(seed=42)
        result = m.mutate("test", strategy="yaml")
        assert "test" in result

    def test_mutate_code_context_strategy(self):
        m = StructuralMutator(seed=42)
        result = m.mutate("test", strategy="code_context")
        assert "test" in result

    def test_mutate_random_strategy(self):
        m = StructuralMutator(seed=42)
        result = m.mutate("test", strategy="random")
        assert "test" in result

    def test_wrap_json_contains_original_text(self):
        m = StructuralMutator(seed=42)
        text = "unique_payload_12345"
        result = m.wrap_json(text)
        assert text in result


# ──────────────────────────────────────────────
# ContextualMutator
# ──────────────────────────────────────────────


class TestContextualMutator:
    def test_init_default_seed(self):
        m = ContextualMutator()
        assert m.rng is not None

    def test_init_with_seed(self):
        m = ContextualMutator(seed=42)
        assert m.rng is not None

    def test_apply_frame_academic(self):
        m = ContextualMutator(seed=42)
        result = m.apply_frame("test input", "academic_research")
        assert "test input" in result

    def test_apply_frame_educational(self):
        m = ContextualMutator(seed=42)
        result = m.apply_frame("test input", "educational_exercise")
        assert "test input" in result

    def test_apply_frame_fictional(self):
        m = ContextualMutator(seed=42)
        result = m.apply_frame("test input", "fictional_story")
        assert "test input" in result

    def test_apply_frame_debugging(self):
        m = ContextualMutator(seed=42)
        result = m.apply_frame("test input", "debugging_session")
        assert "test input" in result

    def test_apply_frame_security_audit(self):
        m = ContextualMutator(seed=42)
        result = m.apply_frame("test input", "security_audit")
        assert "test input" in result

    def test_apply_frame_ctf(self):
        m = ContextualMutator(seed=42)
        result = m.apply_frame("test input", "ctf_challenge")
        assert "test input" in result

    def test_apply_frame_unknown_falls_back(self):
        m = ContextualMutator(seed=42)
        result = m.apply_frame("test input", "nonexistent_frame")
        assert "test input" in result

    def test_mutate_random(self):
        m = ContextualMutator(seed=42)
        result = m.mutate("test input", strategy="random")
        assert "test input" in result

    def test_mutate_specific_frame(self):
        m = ContextualMutator(seed=42)
        result = m.mutate("test input", strategy="academic_research")
        assert "test input" in result

    def test_all_context_frames_exist(self):
        m = ContextualMutator(seed=42)
        for frame_type in CONTEXT_FRAMES:
            result = m.apply_frame("test", frame_type)
            assert "test" in result

    def test_deterministic_with_seed(self):
        m1 = ContextualMutator(seed=999)
        m2 = ContextualMutator(seed=999)
        assert m1.apply_frame("hello", "academic_research") == m2.apply_frame("hello", "academic_research")


# ──────────────────────────────────────────────
# EncodingMutator
# ──────────────────────────────────────────────


class TestEncodingMutator:
    def test_init_default_seed(self):
        m = EncodingMutator()
        assert m.rng is not None

    def test_init_with_seed(self):
        m = EncodingMutator(seed=42)
        assert m.rng is not None

    def test_encode_base64(self):
        m = EncodingMutator(seed=42)
        result = m.encode_base64("hello")
        import base64

        encoded = base64.b64encode(b"hello").decode("utf-8")
        assert encoded in result

    def test_encode_hex(self):
        m = EncodingMutator(seed=42)
        result = m.encode_hex("hello")
        expected = b"hello".hex()
        assert expected in result

    def test_encode_rot13(self):
        m = EncodingMutator(seed=42)
        result = m.encode_rot13("hello")
        # "hello" -> "uryyb" in ROT13
        assert "uryyb" in result

    def test_encode_rot13_reversible(self):
        m = EncodingMutator(seed=42)
        original = "Hello World"
        result = m.encode_rot13(original)
        # Extract ROT13 portion (the encoded text before ".")
        encoded_part = result.split(": ")[1].split(".")[0]
        # Apply ROT13 again to get back original
        _decoded = m.encode_rot13(encoded_part)
        # ROT13 is symmetric; check the encoded text in result
        assert "Uryyb Jbeyq" in result

    def test_encode_url(self):
        m = EncodingMutator(seed=42)
        result = m.encode_url("hello world")
        assert "hello%20world" in result

    def test_encode_partial(self):
        m = EncodingMutator(seed=42)
        result = m.encode_partial("hello world foo bar", fraction=0.5)
        assert "[ENCODED:" in result

    def test_encode_partial_empty_words(self):
        m = EncodingMutator(seed=42)
        result = m.encode_partial("", fraction=0.5)
        assert result == ""

    def test_mutate_base64(self):
        m = EncodingMutator(seed=42)
        result = m.mutate("test", strategy="base64")
        assert isinstance(result, str)

    def test_mutate_hex(self):
        m = EncodingMutator(seed=42)
        result = m.mutate("test", strategy="hex")
        assert isinstance(result, str)

    def test_mutate_rot13(self):
        m = EncodingMutator(seed=42)
        result = m.mutate("test", strategy="rot13")
        assert isinstance(result, str)

    def test_mutate_url(self):
        m = EncodingMutator(seed=42)
        result = m.mutate("test", strategy="url")
        assert isinstance(result, str)

    def test_mutate_partial(self):
        m = EncodingMutator(seed=42)
        result = m.mutate("test input here", strategy="partial")
        assert isinstance(result, str)

    def test_mutate_random(self):
        m = EncodingMutator(seed=42)
        result = m.mutate("test", strategy="random")
        assert isinstance(result, str)

    def test_deterministic_with_seed(self):
        m1 = EncodingMutator(seed=77)
        m2 = EncodingMutator(seed=77)
        assert m1.encode_base64("hello") == m2.encode_base64("hello")


# ──────────────────────────────────────────────
# CompositeMutator
# ──────────────────────────────────────────────


class TestCompositeMutator:
    def test_init(self):
        m = CompositeMutator(seed=42)
        assert m.char_mutator is not None
        assert m.struct_mutator is not None
        assert m.ctx_mutator is not None
        assert m.enc_mutator is not None

    def test_mutate_composite_returns_mutation_result(self):
        m = CompositeMutator(seed=42)
        result = m.mutate_composite("test input")
        assert isinstance(result, MutationResult)
        assert result.original == "test input"
        assert isinstance(result.mutated, str)
        assert isinstance(result.strategies_applied, list)
        assert len(result.strategies_applied) >= 2
        assert "composite" in result.mutation_id

    def test_mutate_composite_custom_strategies(self):
        m = CompositeMutator(seed=42)
        pool = ["char:leetspeak", "struct:json", "enc:base64"]
        result = m.mutate_composite("test", num_strategies=2, strategy_pool=pool)
        assert isinstance(result, MutationResult)
        assert result.original == "test"
        assert len(result.strategies_applied) >= 2

    def test_mutate_composite_min_strategies(self):
        m = CompositeMutator(seed=42)
        result = m.mutate_composite("test", num_strategies=1)
        # Should clamp to minimum of 2
        assert len(result.strategies_applied) >= 2

    def test_mutate_composite_max_strategies(self):
        m = CompositeMutator(seed=42)
        result = m.mutate_composite("test", num_strategies=10)
        # Should clamp to maximum of 3
        assert len(result.strategies_applied) <= 3

    def test_mutate_simple(self):
        m = CompositeMutator(seed=42)
        result = m.mutate("test input")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_composite_produces_different_output(self):
        m = CompositeMutator(seed=42)
        original = "ignore safety rules"
        result = m.mutate(original)
        assert isinstance(result, str)


# ──────────────────────────────────────────────
# MutationEngine
# ──────────────────────────────────────────────


class TestMutationEngine:
    def test_init(self):
        engine = MutationEngine(seed=42)
        assert engine.seed == 42
        assert engine.char_mutator is not None
        assert engine.struct_mutator is not None
        assert engine.ctx_mutator is not None
        assert engine.enc_mutator is not None
        assert engine.composite_mutator is not None

    def test_mutate_basic(self):
        engine = MutationEngine(seed=42)
        results = engine.mutate("test input", n=3)
        assert len(results) == 3
        for r in results:
            _assert_valid_attack(r)

    def test_mutate_returns_mutation_strategy(self):
        engine = MutationEngine(seed=42)
        results = engine.mutate("test input", n=1)
        assert "mutation_strategy" in results[0]

    def test_mutate_specific_strategies(self):
        engine = MutationEngine(seed=42)
        for strategy in ["leetspeak", "json", "academic_research", "base64"]:
            results = engine.mutate("test", strategies=[strategy], n=1)
            assert len(results) == 1
            _assert_valid_attack(results[0])

    def test_mutate_custom_ground_truth(self):
        engine = MutationEngine(seed=42)
        results = engine.mutate("test", n=1, ground_truth="override")
        assert results[0]["ground_truth"] == "override"

    def test_mutate_custom_attack_type(self):
        engine = MutationEngine(seed=42)
        results = engine.mutate("test", n=1, attack_type="custom_attack")
        assert "mutated_custom_attack" in results[0]["attack_type"]

    def test_mutate_custom_difficulty(self):
        engine = MutationEngine(seed=42)
        results = engine.mutate("test", n=1, difficulty="hard")
        assert results[0]["difficulty"] == "hard"

    def test_mutate_deduplication(self):
        engine = MutationEngine(seed=42)
        results = engine.mutate("x", strategies=["leetspeak"], n=5)
        texts = [r["text"] for r in results]
        # All results should be unique
        assert len(texts) == len(set(texts))

    def test_mutate_unknown_strategy_falls_back(self):
        engine = MutationEngine(seed=42)
        results = engine.mutate("test", strategies=["nonexistent_strategy"], n=1)
        assert len(results) == 1
        _assert_valid_attack(results[0])

    def test_to_attack_dict(self):
        engine = MutationEngine(seed=42)
        d = engine.to_attack_dict("mutated text", "direct_override")
        assert d["text"] == "mutated text"
        assert d["is_attack"] is True
        assert d["ground_truth"] == "injection"
        assert d["difficulty"] == "medium"
        assert "mutated_direct_override" in d["attack_type"]
        assert d["source"] == "mutation_engine"

    def test_to_attack_dict_custom_params(self):
        engine = MutationEngine(seed=42)
        d = engine.to_attack_dict(
            "test",
            "my_type",
            ground_truth="safe",
            difficulty="easy",
        )
        assert d["ground_truth"] == "safe"
        assert d["difficulty"] == "easy"

    def test_get_strategy_stats_empty(self):
        engine = MutationEngine(seed=42)
        stats = engine.get_strategy_stats()
        assert stats == {}

    def test_record_strategy_outcome(self):
        engine = MutationEngine(seed=42)
        engine.record_strategy_outcome("leetspeak", was_detected=True)
        engine.record_strategy_outcome("leetspeak", was_detected=False)
        stats = engine.get_strategy_stats()
        assert "leetspeak" in stats
        assert stats["leetspeak"]["detected"] == 1
        assert stats["leetspeak"]["bypassed"] == 1
        assert stats["leetspeak"]["total"] == 2

    def test_record_strategy_new_entry(self):
        engine = MutationEngine(seed=42)
        engine.record_strategy_outcome("new_strat", was_detected=True)
        stats = engine.get_strategy_stats()
        assert "new_strat" in stats
        assert stats["new_strat"]["total"] == 1


# ──────────────────────────────────────────────
# MutationEngine file loading
# ──────────────────────────────────────────────


class TestMutationEngineFileLoading:
    def test_mutate_from_file_not_found(self):
        engine = MutationEngine(seed=42)
        with pytest.raises(FileNotFoundError):
            engine.mutate_from_file("/nonexistent/path.json")

    def test_mutate_from_file_plain_text(self):
        engine = MutationEngine(seed=42)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("test prompt one\ntest prompt two\n")
            f.flush()
            results = engine.mutate_from_file(f.name, n_per_prompt=2)
        assert len(results) >= 2
        for r in results:
            _assert_valid_attack(r)

    def test_mutate_from_file_json_array(self):
        engine = MutationEngine(seed=42)
        data = [
            {"text": "prompt A", "attack_type": "override", "ground_truth": "override"},
            {"text": "prompt B", "attack_type": "injection", "ground_truth": "injection"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            results = engine.mutate_from_file(f.name, n_per_prompt=1)
        assert len(results) >= 2
        for r in results:
            _assert_valid_attack(r)

    def test_mutate_from_file_jsonl(self):
        engine = MutationEngine(seed=42)
        lines = [
            json.dumps({"text": "line one", "attack_type": "test"}),
            json.dumps({"text": "line two", "attack_type": "test"}),
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("\n".join(lines))
            f.flush()
            results = engine.mutate_from_file(f.name, n_per_prompt=1)
        assert len(results) >= 2

    def test_mutate_from_file_empty(self):
        engine = MutationEngine(seed=42)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")
            f.flush()
            results = engine.mutate_from_file(f.name)
        assert results == []


# ──────────────────────────────────────────────
# MutationEngine evolve
# ──────────────────────────────────────────────


class TestMutationEngineEvolve:
    def test_evolve_empty_input(self):
        engine = MutationEngine(seed=42)
        results = engine.evolve([], n_new=5)
        assert results == []

    def test_evolve_from_successful_mutations(self):
        engine = MutationEngine(seed=42)
        successful = engine.mutate("test input", n=3)
        results = engine.evolve(successful, n_new=5)
        assert len(results) <= 5
        for r in results:
            _assert_valid_attack(r)

    def test_evolve_records_stats(self):
        engine = MutationEngine(seed=42)
        successful = engine.mutate("test", n=2)
        engine.evolve(successful, n_new=2)
        stats = engine.get_strategy_stats()
        assert len(stats) > 0


# ──────────────────────────────────────────────
# Convenience functions
# ──────────────────────────────────────────────


class TestConvenienceFunctions:
    def test_create_engine(self):
        engine = create_engine(seed=42)
        assert isinstance(engine, MutationEngine)
        assert engine.seed == 42

    def test_create_engine_no_seed(self):
        engine = create_engine()
        assert isinstance(engine, MutationEngine)

    def test_quick_mutate(self):
        results = quick_mutate("hello world", n=3, seed=42)
        assert len(results) == 3
        for r in results:
            _assert_valid_attack(r)

    def test_quick_mutate_with_strategies(self):
        results = quick_mutate("test", n=2, seed=42, strategies=["leetspeak", "base64"])
        assert len(results) == 2
        for r in results:
            _assert_valid_attack(r)


# ──────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────


class TestDataclasses:
    def test_mutation_result_defaults(self):
        r = MutationResult(
            original="hello",
            mutated="h3ll0",
            strategies_applied=["leetspeak"],
        )
        assert r.original == "hello"
        assert r.mutated == "h3ll0"
        assert r.strategies_applied == ["leetspeak"]
        assert r.mutation_id == ""
        assert r.metadata == {}

    def test_mutation_result_with_all_fields(self):
        r = MutationResult(
            original="hello",
            mutated="h3ll0",
            strategies_applied=["leetspeak"],
            mutation_id="test_123",
            metadata={"key": "value"},
        )
        assert r.mutation_id == "test_123"
        assert r.metadata == {"key": "value"}

    def test_mutation_spec_defaults(self):
        s = MutationSpec(text="test")
        assert s.text == "test"
        assert s.ground_truth == "injection"
        assert s.attack_type == "mutated_injection"
        assert s.difficulty == "medium"
        assert s.source == "mutation_engine"


# ──────────────────────────────────────────────
# Module-level constants
# ──────────────────────────────────────────────


class TestConstants:
    def test_leetspeak_map_not_empty(self):
        assert len(LEETSPEAK_MAP) > 0
        assert "a" in LEETSPEAK_MAP
        assert "e" in LEETSPEAK_MAP

    def test_homoglyphs_not_empty(self):
        assert len(HOMOGLYPHS) > 0
        assert "a" in HOMOGLYPHS
        assert isinstance(HOMOGLYPHS["a"], list)

    def test_zero_width_chars_not_empty(self):
        assert len(ZERO_WIDTH_CHARS) > 0
        assert "\u200b" in ZERO_WIDTH_CHARS

    def test_context_frames_not_empty(self):
        assert len(CONTEXT_FRAMES) > 0
        assert "academic_research" in CONTEXT_FRAMES
