#!/usr/bin/env python3
"""
Tests for df.textproc — the deterministic cleanup pipeline.

Structure mirrors the module: one TestCase per public function, plus a class
for the orchestrator and one for pathological inputs. Every false-positive case
named in the module's docstrings has a test here; those are the assertions that
matter most, because a transformation that fails to fire is invisible while one
that fires wrongly silently rewrites what the user said.

Run:  .venv/bin/python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from df.textproc import (  # noqa: E402
    CLEANUP_LEVELS,
    apply_backtrack,
    apply_dictionary,
    apply_spoken_punctuation,
    detect_trailing_command,
    expand_snippets,
    format_lists,
    normalize_spacing,
    process,
    remove_fillers,
)


class TestApplyDictionary(unittest.TestCase):
    def test_empty_entries_is_identity(self):
        self.assertEqual(apply_dictionary("hello world", []), "hello world")

    def test_simple_replacement(self):
        entries = [{"from": "clod", "to": "Claude"}]
        self.assertEqual(apply_dictionary("ask clod about it", entries), "ask Claude about it")

    def test_case_insensitive_match(self):
        entries = [{"from": "clod", "to": "Claude"}]
        self.assertEqual(apply_dictionary("ask CLOD", entries), "ask CLAUDE")

    def test_whole_word_only(self):
        entries = [{"from": "API", "to": "endpoint"}]
        # "APIs" must survive untouched — no entry covers the plural. The
        # match itself is ALL CAPS, so the replacement inherits that.
        self.assertEqual(apply_dictionary("the API and the APIs", entries),
                         "the ENDPOINT and the APIs")

    def test_whole_word_only_lowercase(self):
        entries = [{"from": "api", "to": "endpoint"}]
        self.assertEqual(apply_dictionary("the api and the apis", entries),
                         "the endpoint and the apis")

    def test_no_match_inside_a_longer_word(self):
        entries = [{"from": "cat", "to": "dog"}]
        self.assertEqual(apply_dictionary("concatenate the catalog", entries),
                         "concatenate the catalog")

    def test_capitalized_match_capitalizes_replacement(self):
        entries = [{"from": "sofie", "to": "Sophie"}]
        self.assertEqual(apply_dictionary("Sofie called", entries), "Sophie called")
        entries = [{"from": "clod", "to": "claude"}]
        self.assertEqual(apply_dictionary("Clod said hi", entries), "Claude said hi")

    def test_all_caps_match_uppercases_replacement(self):
        entries = [{"from": "asap", "to": "as soon as possible"}]
        self.assertEqual(apply_dictionary("send it ASAP", entries),
                         "send it AS SOON AS POSSIBLE")

    def test_single_capital_letter_is_not_all_caps(self):
        # "I" is one letter; treating it as ALL CAPS would upper the whole
        # replacement, which is never what the user meant.
        entries = [{"from": "i", "to": "eye"}]
        self.assertEqual(apply_dictionary("I see", entries), "Eye see")

    def test_lowercase_match_keeps_replacement_casing(self):
        entries = [{"from": "github", "to": "GitHub"}]
        self.assertEqual(apply_dictionary("push to github now", entries),
                         "push to GitHub now")

    def test_multi_word_from(self):
        entries = [{"from": "pull request", "to": "PR"}]
        self.assertEqual(apply_dictionary("open a pull request today", entries),
                         "open a PR today")

    def test_multi_word_from_tolerates_extra_whitespace(self):
        entries = [{"from": "pull request", "to": "PR"}]
        self.assertEqual(apply_dictionary("open a pull  request", entries), "open a PR")

    def test_longest_from_wins(self):
        entries = [
            {"from": "machine", "to": "MACHINE"},
            {"from": "machine learning", "to": "ML"},
        ]
        self.assertEqual(apply_dictionary("machine learning is fun", entries),
                         "ML is fun")

    def test_single_pass_does_not_reapply_rules(self):
        # "as" must not be rewritten inside the text produced by the "asap" rule.
        entries = [{"from": "asap", "to": "as soon as possible"},
                   {"from": "as", "to": "AS"}]
        self.assertEqual(apply_dictionary("do it asap", entries),
                         "do it as soon as possible")

    def test_regex_metacharacters_in_from_are_escaped(self):
        entries = [{"from": "C++", "to": "C plus plus"}]
        self.assertEqual(apply_dictionary("I write C++ code", entries),
                         "I write C plus plus code")

    def test_regex_dot_is_literal_not_wildcard(self):
        entries = [{"from": "a.b", "to": "MATCH"}]
        # "arb" would match if "." were left as a regex wildcard.
        self.assertEqual(apply_dictionary("arb and a.b and axb", entries),
                         "arb and MATCH and axb")

    def test_regex_metacharacter_soup_does_not_raise(self):
        entries = [{"from": "(foo)*", "to": "bar"}, {"from": "[a-z]", "to": "baz"}]
        self.assertEqual(apply_dictionary("call (foo)* and [a-z] now", entries),
                         "call bar and baz now")

    def test_cpp_at_end_of_string(self):
        # \b would fail here because "+" is not a word character.
        entries = [{"from": "C++", "to": "C plus plus"}]
        self.assertEqual(apply_dictionary("I know C++", entries), "I know C plus plus")

    def test_replacement_at_start_and_end(self):
        entries = [{"from": "foo", "to": "bar"}]
        self.assertEqual(apply_dictionary("foo middle foo", entries), "bar middle bar")

    def test_blank_from_is_ignored(self):
        entries = [{"from": "", "to": "x"}, {"from": "   ", "to": "y"}]
        self.assertEqual(apply_dictionary("untouched", entries), "untouched")

    def test_malformed_entries_are_skipped(self):
        # A half-written entry must not become a deletion rule.
        entries = ["not a dict", {"from": "a"}, {"to": "b"}, 42, None]
        self.assertEqual(apply_dictionary("a b", entries), "a b")

    def test_explicit_empty_to_is_a_deletion(self):
        self.assertEqual(apply_dictionary("say foo bar", [{"from": "foo", "to": ""}]),
                         "say  bar")

    def test_empty_text(self):
        self.assertEqual(apply_dictionary("", [{"from": "a", "to": "b"}]), "")


class TestExpandSnippets(unittest.TestCase):
    def test_expansion(self):
        snips = [{"trigger": "sig", "text": "Best,\nSophie"}]
        self.assertEqual(expand_snippets("thanks sig", snips), "thanks Best,\nSophie")

    def test_case_insensitive_trigger(self):
        snips = [{"trigger": "sig", "text": "Best, Sophie"}]
        self.assertEqual(expand_snippets("thanks SIG", snips), "thanks Best, Sophie")

    def test_snippet_text_inserted_verbatim(self):
        # No case-style carry-over: a canned block keeps its own capitalization.
        snips = [{"trigger": "addr", "text": "1 Main St"}]
        self.assertEqual(expand_snippets("ADDR", snips), "1 Main St")

    def test_mid_sentence(self):
        snips = [{"trigger": "eod", "text": "end of day"}]
        self.assertEqual(expand_snippets("due eod tomorrow", snips), "due end of day tomorrow")

    def test_not_glued_to_word_characters(self):
        snips = [{"trigger": "sig", "text": "SIGNATURE"}]
        self.assertEqual(expand_snippets("sigma design", snips), "sigma design")
        self.assertEqual(expand_snippets("resig", snips), "resig")

    def test_trigger_adjacent_to_punctuation_still_matches(self):
        snips = [{"trigger": "sig", "text": "Best"}]
        self.assertEqual(expand_snippets("thanks, sig.", snips), "thanks, Best.")

    def test_multi_word_trigger(self):
        snips = [{"trigger": "my address", "text": "1 Main St"}]
        self.assertEqual(expand_snippets("send to my address please", snips),
                         "send to 1 Main St please")

    def test_longest_trigger_first(self):
        snips = [{"trigger": "sig", "text": "SHORT"}, {"trigger": "sig block", "text": "LONG"}]
        self.assertEqual(expand_snippets("add sig block here", snips), "add LONG here")

    def test_single_pass(self):
        snips = [{"trigger": "a", "text": "b"}, {"trigger": "b", "text": "c"}]
        self.assertEqual(expand_snippets("a", snips), "b")

    def test_empty_list(self):
        self.assertEqual(expand_snippets("hello", []), "hello")


class TestSpokenPunctuation(unittest.TestCase):
    def test_period_at_end(self):
        self.assertEqual(apply_spoken_punctuation("I am done period"), "I am done.")

    def test_full_stop_at_end(self):
        self.assertEqual(apply_spoken_punctuation("I am done full stop"), "I am done.")

    def test_period_before_capitalized_word(self):
        self.assertEqual(
            apply_spoken_punctuation("I went to the store period Then I came home"),
            "I went to the store. Then I came home",
        )

    def test_comma_before_clause_starter(self):
        self.assertEqual(
            apply_spoken_punctuation("send the report comma then call Dana"),
            "send the report, then call Dana",
        )

    def test_question_mark(self):
        self.assertEqual(apply_spoken_punctuation("are you free question mark"),
                         "are you free?")

    def test_exclamation_mark_and_point(self):
        self.assertEqual(apply_spoken_punctuation("that is great exclamation mark"),
                         "that is great!")
        self.assertEqual(apply_spoken_punctuation("that is great exclamation point"),
                         "that is great!")

    def test_colon_and_semicolon(self):
        self.assertEqual(apply_spoken_punctuation("here it is colon it works"),
                         "here it is: it works")
        self.assertEqual(apply_spoken_punctuation("it broke semicolon we fixed it"),
                         "it broke; we fixed it")

    def test_parens_wrap_content(self):
        self.assertEqual(
            apply_spoken_punctuation("the total open paren before tax close paren is 40"),
            "the total (before tax) is 40",
        )

    def test_parenthesis_long_form(self):
        self.assertEqual(
            apply_spoken_punctuation("cost open parenthesis net close parenthesis rose"),
            "cost (net) rose",
        )

    def test_quote_and_unquote(self):
        self.assertEqual(
            apply_spoken_punctuation("he said quote I am tired unquote and left"),
            'he said "I am tired" and left',
        )

    def test_dash_is_spaced_em_dash(self):
        self.assertEqual(apply_spoken_punctuation("the plan is simple dash we ship Friday"),
                         "the plan is simple — we ship Friday")

    def test_dash_not_converted_before_an_ordinary_word(self):
        # "mostly" doesn't start a clause, so the command reading is unproven.
        self.assertEqual(apply_spoken_punctuation("it works dash mostly"),
                         "it works dash mostly")

    def test_hyphen_joins_without_spaces(self):
        self.assertEqual(apply_spoken_punctuation("state hyphen Of the art"),
                         "state-Of the art")

    def test_ellipsis(self):
        self.assertEqual(apply_spoken_punctuation("and then ellipsis"), "and then…")

    def test_new_line(self):
        self.assertEqual(apply_spoken_punctuation("first item new line second item"),
                         "first item\nsecond item")
        self.assertEqual(apply_spoken_punctuation("first item newline second item"),
                         "first item\nsecond item")

    def test_new_paragraph(self):
        self.assertEqual(
            apply_spoken_punctuation("the meeting is at noon new paragraph Please bring notes"),
            "the meeting is at noon\n\nPlease bring notes",
        )

    def test_two_commands_in_a_row(self):
        self.assertEqual(
            apply_spoken_punctuation("he said open quote hello close quote period"),
            'he said "hello".',
        )

    def test_bare_quote_keeps_the_strict_test(self):
        # Unlike "open quote", bare "quote" is a real English noun, so it only
        # converts when what follows clearly starts a clause.
        self.assertEqual(apply_spoken_punctuation("he said quote hello there"),
                         "he said quote hello there")

    # ── false positives: these must NOT convert ────────────────
    def test_no_convert_a_period_of_time(self):
        self.assertEqual(apply_spoken_punctuation("a period of time"), "a period of time")
        self.assertEqual(apply_spoken_punctuation("it lasted a period of time"),
                         "it lasted a period of time")

    def test_no_convert_the_comma_operator(self):
        self.assertEqual(apply_spoken_punctuation("the comma operator is weird"),
                         "the comma operator is weird")

    def test_no_convert_period_costume(self):
        self.assertEqual(apply_spoken_punctuation("period costume"), "period costume")
        self.assertEqual(apply_spoken_punctuation("she wore a period costume"),
                         "she wore a period costume")

    def test_no_convert_during_that_period(self):
        self.assertEqual(apply_spoken_punctuation("during that period"),
                         "during that period")

    def test_no_convert_at_start_of_input(self):
        self.assertEqual(apply_spoken_punctuation("period"), "period")
        self.assertEqual(apply_spoken_punctuation("comma separated values"),
                         "comma separated values")

    def test_no_convert_when_next_word_continues_the_phrase(self):
        self.assertEqual(apply_spoken_punctuation("use comma separated values"),
                         "use comma separated values")

    def test_no_convert_a_new_line_of_code(self):
        self.assertEqual(apply_spoken_punctuation("add a new line of code"),
                         "add a new line of code")

    def test_no_convert_famous_quote(self):
        self.assertEqual(apply_spoken_punctuation("that is a famous quote"),
                         "that is a famous quote")

    def test_empty_and_whitespace(self):
        self.assertEqual(apply_spoken_punctuation(""), "")
        self.assertEqual(apply_spoken_punctuation("   "), "   ")

    def test_multiline_input_is_processed_per_line(self):
        self.assertEqual(
            apply_spoken_punctuation("I am done period\nnext line here"),
            "I am done.\nnext line here",
        )


class TestBacktrack(unittest.TestCase):
    def test_documented_wispr_example(self):
        self.assertEqual(apply_backtrack("Let's do coffee at 2 actually 3"),
                         "Let's do coffee at 3")

    def test_bare_actually_is_left_alone(self):
        # No comparable-token swap available — this is an intensifier.
        self.assertEqual(apply_backtrack("I actually think that's right"),
                         "I actually think that's right")

    def test_time_swap(self):
        self.assertEqual(apply_backtrack("the call is at 3:30 actually 4:00"),
                         "the call is at 4:00")

    def test_weekday_swap(self):
        self.assertEqual(apply_backtrack("let's meet Tuesday actually Wednesday"),
                         "let's meet Wednesday")

    def test_month_swap(self):
        self.assertEqual(apply_backtrack("it ships in June actually July"),
                         "it ships in July")

    def test_mismatched_classes_do_not_swap(self):
        self.assertEqual(apply_backtrack("we met Tuesday actually 4"),
                         "we met Tuesday actually 4")

    def test_scratch_that_drops_the_sentence_so_far(self):
        self.assertEqual(apply_backtrack("Let's meet at 3, scratch that, let's meet at 4"),
                         "let's meet at 4")

    def test_scratch_that_respects_earlier_sentences(self):
        self.assertEqual(
            apply_backtrack("I finished the report. Scratch that, I finished the draft"),
            "I finished the report. I finished the draft",
        )

    def test_no_wait_trigger(self):
        self.assertEqual(apply_backtrack("send it to Dana no wait send it to Sam"),
                         "send it to Sam")

    def test_i_mean_trigger(self):
        self.assertEqual(apply_backtrack("the deadline is Friday I mean Monday"),
                         "Monday")

    def test_actually_no_trigger(self):
        self.assertEqual(apply_backtrack("book the 9am actually no book the 10am"),
                         "book the 10am")

    def test_correction_trigger(self):
        self.assertEqual(apply_backtrack("it costs 50 correction it costs 60"),
                         "it costs 60")

    def test_trigger_is_case_insensitive(self):
        self.assertEqual(apply_backtrack("plan A. SCRATCH THAT, plan B is better"),
                         "plan A. plan B is better")

    def test_trigger_with_nothing_after_is_kept(self):
        # Deleting the clause would leave an empty utterance.
        self.assertEqual(apply_backtrack("scratch that"), "scratch that")
        self.assertEqual(apply_backtrack("send the note, scratch that."),
                         "send the note, scratch that.")

    def test_word_stutter(self):
        self.assertEqual(apply_backtrack("the the report"), "the report")
        self.assertEqual(apply_backtrack("I I think"), "I think")

    def test_stutter_is_case_insensitive(self):
        self.assertEqual(apply_backtrack("The the report is done"), "The report is done")

    def test_triple_stutter(self):
        self.assertEqual(apply_backtrack("we we we should go"), "we should go")

    def test_non_adjacent_repeat_is_kept(self):
        self.assertEqual(apply_backtrack("the report and the summary"),
                         "the report and the summary")

    def test_digits_are_not_collapsed(self):
        self.assertEqual(apply_backtrack("version 1 1 shipped"), "version 1 1 shipped")

    def test_empty_input(self):
        self.assertEqual(apply_backtrack(""), "")
        self.assertEqual(apply_backtrack("  "), "  ")


class TestRemoveFillers(unittest.TestCase):
    def test_level_none_is_identity(self):
        self.assertEqual(remove_fillers("um, I like, you know, so basically", "none"),
                         "um, I like, you know, so basically")

    def test_invalid_level_raises(self):
        with self.assertRaises(ValueError):
            remove_fillers("hi", "aggressive")

    def test_light_removes_hesitations(self):
        self.assertEqual(remove_fillers("Um, I think, uh, we should go", "light"),
                         "I think we should go")

    def test_light_variants(self):
        # Casing is left alone when the input didn't start with a capital —
        # normalize_spacing owns sentence capitals, not this function.
        self.assertEqual(remove_fillers("erm let me check", "light"), "let me check")
        self.assertEqual(remove_fillers("hmm ummm uhh okay", "light"), "okay")

    def test_light_restores_a_leading_capital(self):
        self.assertEqual(remove_fillers("Um the report is done", "light"),
                         "The report is done")

    def test_light_leaves_discourse_fillers(self):
        self.assertEqual(remove_fillers("It was, like, good", "light"), "It was, like, good")

    def test_light_does_not_eat_millimetres(self):
        # "5 mm" is a measurement, not a hesitation.
        self.assertEqual(remove_fillers("The gap is 5 mm wide", "light"),
                         "The gap is 5 mm wide")

    def test_light_does_not_match_inside_words(self):
        self.assertEqual(remove_fillers("umbrella under the summer sun", "light"),
                         "umbrella under the summer sun")

    def test_medium_removes_parenthetical_like(self):
        self.assertEqual(remove_fillers("It was, like, really good", "medium"),
                         "It was really good")

    def test_medium_keeps_meaningful_like(self):
        self.assertEqual(remove_fillers("I like coffee", "medium"), "I like coffee")
        self.assertEqual(remove_fillers("It looks like rain", "medium"), "It looks like rain")
        self.assertEqual(remove_fillers("like a boss", "medium"), "like a boss")
        self.assertEqual(remove_fillers("do it like this", "medium"), "do it like this")

    def test_medium_removes_you_know(self):
        self.assertEqual(remove_fillers("It's, you know, complicated", "medium"),
                         "It's complicated")

    def test_medium_keeps_meaningful_you_know(self):
        self.assertEqual(remove_fillers("I wonder if you know the answer", "medium"),
                         "I wonder if you know the answer")

    def test_medium_removes_sentence_initial_i_mean(self):
        self.assertEqual(remove_fillers("I mean, it's fine", "medium"), "It's fine")

    def test_medium_keeps_meaningful_i_mean(self):
        self.assertEqual(remove_fillers("You know what I mean here", "medium"),
                         "You know what I mean here")

    def test_medium_removes_hedges(self):
        self.assertEqual(remove_fillers("It was kind of weird", "medium"), "It was weird")
        self.assertEqual(remove_fillers("I was sort of tired", "medium"), "I was tired")

    def test_medium_keeps_determined_hedges(self):
        self.assertEqual(remove_fillers("What kind of car is that", "medium"),
                         "What kind of car is that")
        self.assertEqual(remove_fillers("It is a sort of hybrid", "medium"),
                         "It is a sort of hybrid")

    def test_medium_removes_basically_and_literally(self):
        self.assertEqual(remove_fillers("I basically just literally ran", "medium"),
                         "I just ran")

    def test_medium_includes_light(self):
        self.assertEqual(remove_fillers("Um, it was, like, fine", "medium"), "It was fine")

    def test_high_removes_sentence_openers(self):
        self.assertEqual(remove_fillers("So, the answer is 42", "high"), "The answer is 42")
        self.assertEqual(remove_fillers("Well I disagree", "high"), "I disagree")
        self.assertEqual(remove_fillers("Okay let's start", "high"), "Let's start")

    def test_high_keeps_mid_sentence_openers(self):
        self.assertEqual(remove_fillers("I said so and I meant it", "high"),
                         "I said so and I meant it")

    def test_high_removes_tag_question(self):
        self.assertEqual(remove_fillers("That's the plan, right?", "high"),
                         "That's the plan.")
        self.assertEqual(remove_fillers("We ship Friday, you know?", "high"),
                         "We ship Friday.")

    def test_high_keeps_real_question(self):
        # No comma before "right" — this is a genuine question.
        self.assertEqual(remove_fillers("Is that right?", "high"), "Is that right?")

    def test_high_includes_medium_and_light(self):
        self.assertEqual(remove_fillers("So, um, it was, like, fine", "high"), "It was fine")

    def test_spacing_and_commas_are_tidied(self):
        self.assertEqual(remove_fillers("I think um , uh , we go", "light"), "I think we go")

    def test_empty_input_at_every_level(self):
        for level in CLEANUP_LEVELS:
            self.assertEqual(remove_fillers("", level), "")
            self.assertEqual(remove_fillers("   ", level), "   ")


class TestFormatLists(unittest.TestCase):
    def test_documented_example(self):
        self.assertEqual(
            format_lists(
                "My top goals this week are one finish the report "
                "two send the presentation three call Dana"
            ),
            "My top goals this week are:\n"
            "1. Finish the report\n"
            "2. Send the presentation\n"
            "3. Call Dana",
        )

    def test_ordinal_words(self):
        self.assertEqual(
            format_lists("The plan is first draft the memo second review the memo"),
            "The plan is:\n1. Draft the memo\n2. Review the memo",
        )

    def test_two_items_is_enough(self):
        self.assertEqual(
            format_lists("Todo one buy milk today two walk the dog"),
            "Todo:\n1. Buy milk today\n2. Walk the dog",
        )

    def test_existing_colon_is_not_doubled(self):
        self.assertEqual(
            format_lists("Todo: one buy the milk two walk the dog"),
            "Todo:\n1. Buy the milk\n2. Walk the dog",
        )

    def test_trailing_comma_in_lead_in_is_replaced_by_colon(self):
        self.assertEqual(
            format_lists("Here they are, one buy the milk two walk the dog"),
            "Here they are:\n1. Buy the milk\n2. Walk the dog",
        )

    def test_no_lead_in(self):
        self.assertEqual(
            format_lists("one buy the milk two walk the dog"),
            "1. Buy the milk\n2. Walk the dog",
        )

    def test_items_are_capitalized(self):
        out = format_lists("Steps one open the door two close the door")
        self.assertIn("1. Open the door", out)
        self.assertIn("2. Close the door", out)

    def test_one_bad_marker_cancels_the_whole_list(self):
        # "three apples" is too short to be an item, which is the tell that all
        # three ordinals were quantities. Salvaging items 1 and 2 would turn a
        # shopping sentence into a list.
        self.assertEqual(
            format_lists("Buy one dozen eggs two loaves of bread three apples"),
            "Buy one dozen eggs two loaves of bread three apples",
        )

    # ── false positives ────────────────────────────────────────
    def test_no_list_for_one_of_the_two_options(self):
        self.assertEqual(format_lists("one of the two options"), "one of the two options")
        self.assertEqual(format_lists("pick one of the two options here"),
                         "pick one of the two options here")

    def test_no_list_for_i_have_one_thing_to_say(self):
        self.assertEqual(format_lists("I have one thing to say"), "I have one thing to say")

    def test_no_list_without_two(self):
        self.assertEqual(format_lists("one finish the report and go home"),
                         "one finish the report and go home")

    def test_no_list_when_out_of_order(self):
        self.assertEqual(format_lists("two send the deck one finish the report"),
                         "two send the deck one finish the report")

    def test_no_list_for_durations(self):
        self.assertEqual(format_lists("it takes one hour and two minutes to run"),
                         "it takes one hour and two minutes to run")

    def test_already_multiline_is_untouched(self):
        text = "Todo:\n1. one thing here\n2. two things here"
        self.assertEqual(format_lists(text), text)

    def test_empty_input(self):
        self.assertEqual(format_lists(""), "")


class TestDetectTrailingCommand(unittest.TestCase):
    def test_press_enter(self):
        self.assertEqual(detect_trailing_command("tell her I said hi press enter"),
                         ("tell her I said hi", "enter"))

    def test_press_return(self):
        self.assertEqual(detect_trailing_command("ok press return"), ("ok", "enter"))

    def test_send_it(self):
        self.assertEqual(detect_trailing_command("looks good send it"), ("looks good", "enter"))

    def test_case_insensitive(self):
        self.assertEqual(detect_trailing_command("done PRESS ENTER"), ("done", "enter"))

    def test_trailing_punctuation_allowed(self):
        self.assertEqual(detect_trailing_command("done, press enter."), ("done", "enter"))
        self.assertEqual(detect_trailing_command("done press enter!"), ("done", "enter"))

    def test_trailing_whitespace_allowed(self):
        self.assertEqual(detect_trailing_command("done press enter   "), ("done", "enter"))

    def test_only_at_the_very_end(self):
        self.assertEqual(
            detect_trailing_command("I pressed enter and it sent it to the wrong person"),
            ("I pressed enter and it sent it to the wrong person", None),
        )
        self.assertEqual(detect_trailing_command("press enter then type your name"),
                         ("press enter then type your name", None))

    def test_command_only(self):
        self.assertEqual(detect_trailing_command("send it."), ("", "enter"))

    def test_no_command(self):
        self.assertEqual(detect_trailing_command("hello world"), ("hello world", None))

    def test_empty(self):
        self.assertEqual(detect_trailing_command(""), ("", None))
        self.assertEqual(detect_trailing_command("   "), ("   ", None))


class TestNormalizeSpacing(unittest.TestCase):
    def test_collapses_spaces(self):
        self.assertEqual(normalize_spacing("hello    world there"), "Hello world there.")

    def test_removes_space_before_closing_punctuation(self):
        self.assertEqual(normalize_spacing("hello world ."), "Hello world.")
        self.assertEqual(normalize_spacing("really ? yes !"), "Really? Yes!")

    def test_adds_space_after_sentence_punctuation(self):
        self.assertEqual(normalize_spacing("one,two"), "One, two.")
        # A capital after the dot IS treated as a sentence boundary.
        self.assertEqual(normalize_spacing("one.Two"), "One. Two.")

    def test_lowercase_after_dot_is_treated_as_one_token(self):
        """A dot between two lowercase/alphanumeric characters is left alone.

        This is a deliberate trade-off, not an oversight. "one.two" and
        "gmail.com" are structurally identical, so no local rule can separate
        them — and Whisper effectively never emits "one.two" from speech (it
        produces "one. Two"), while domains, filenames and versions appear in
        dictation constantly. Snippet bodies, which are literal text the user
        typed, made this decisive: an email snippet was being rewritten to
        "arielwalters12@gmail. Com".
        """
        self.assertEqual(normalize_spacing("write to me at a@gmail.com today"),
                         "Write to me at a@gmail.com today.")
        self.assertEqual(normalize_spacing("open file.py now"), "Open file.py now.")
        self.assertEqual(normalize_spacing("we shipped v2.0 today"),
                         "We shipped v2.0 today.")

    def test_decimal_numbers_survive(self):
        self.assertEqual(normalize_spacing("pi is 3.14 exactly"), "Pi is 3.14 exactly.")

    def test_initialisms_survive(self):
        self.assertEqual(normalize_spacing("e.g. this thing"), "E.g. this thing.")
        self.assertEqual(normalize_spacing("the U.S.A. is big"), "The U.S.A. is big.")

    def test_paren_spacing(self):
        self.assertEqual(normalize_spacing("the total ( before tax ) is 40"),
                         "The total (before tax) is 40.")

    def test_capitalizes_each_sentence(self):
        self.assertEqual(normalize_spacing("hello there. how are you? fine!"),
                         "Hello there. How are you? Fine!")

    def test_capitalizes_after_newline(self):
        self.assertEqual(normalize_spacing("first line\nsecond line"),
                         "First line\nSecond line")

    def test_strips_per_line_whitespace(self):
        self.assertEqual(normalize_spacing("  a line  \n   another line  "),
                         "A line\nAnother line")

    def test_collapses_blank_lines(self):
        self.assertEqual(normalize_spacing("a b\n\n\n\n\nc d"), "A b\n\nC d")

    def test_adds_terminal_period(self):
        self.assertEqual(normalize_spacing("this is a sentence"), "This is a sentence.")

    def test_keeps_existing_terminal_punctuation(self):
        self.assertEqual(normalize_spacing("is this a question?"), "Is this a question?")
        self.assertEqual(normalize_spacing("wow that is loud!"), "Wow that is loud!")

    def test_trailing_comma_becomes_period(self):
        self.assertEqual(normalize_spacing("this is a sentence,"), "This is a sentence.")

    def test_no_period_for_single_word(self):
        self.assertEqual(normalize_spacing("hello"), "Hello")
        self.assertEqual(normalize_spacing("  supercalifragilistic  "), "Supercalifragilistic")

    def test_no_period_added_to_multiline_lists(self):
        self.assertEqual(normalize_spacing("Todo:\n1. Buy milk\n2. Walk dog"),
                         "Todo:\n1. Buy milk\n2. Walk dog")

    def test_empty_and_whitespace(self):
        self.assertEqual(normalize_spacing(""), "")
        self.assertEqual(normalize_spacing("   \n  \t "), "")

    def test_windows_newlines(self):
        self.assertEqual(normalize_spacing("a line\r\nb line"), "A line\nB line")

    def test_unicode_is_preserved(self):
        self.assertEqual(normalize_spacing("café  au   lait"), "Café au lait.")

    def test_no_period_appended_after_an_emoji(self):
        self.assertEqual(normalize_spacing("café  au   lait 🎉"), "Café au lait 🎉")


class TestProcess(unittest.TestCase):
    def test_returns_expected_keys(self):
        out = process("hello world")
        self.assertEqual(set(out), {"text", "raw", "command", "changed"})

    def test_raw_is_untouched(self):
        raw = "  um, hello world  "
        self.assertEqual(process(raw)["raw"], raw)

    def test_changed_flag(self):
        self.assertTrue(process("um hello there")["changed"])
        self.assertFalse(process("Hello there.")["changed"])

    def test_invalid_level_raises(self):
        with self.assertRaises(ValueError):
            process("hi", level="turbo")

    def test_trailing_command_is_extracted(self):
        out = process("tell her I said hi press enter")
        self.assertEqual(out["command"], "enter")
        self.assertEqual(out["text"], "Tell her I said hi.")

    def test_no_command_is_none(self):
        self.assertIsNone(process("hello there")["command"])

    def test_backtrack_runs_before_filler_removal(self):
        # "I mean" is both a backtrack trigger and a medium-level filler. If
        # fillers ran first the retracted "Friday" would survive.
        # One word survives, so no terminal period is added (see
        # normalize_spacing) — the point here is that "Friday" is gone.
        out = process("the deadline is Friday I mean Monday", level="medium")
        self.assertEqual(out["text"], "Monday")

    def test_backtrack_can_be_disabled(self):
        out = process("Let's do coffee at 2 actually 3", backtrack=False, level="none")
        self.assertEqual(out["text"], "Let's do coffee at 2 actually 3.")

    def test_level_none_still_does_everything_else(self):
        out = process(
            "um the api is ready period send it",
            level="none",
            dictionary=[{"from": "api", "to": "endpoint"}],
        )
        # Filler kept (level none), but punctuation, dictionary and the
        # trailing command all still ran.
        self.assertEqual(out["command"], "enter")
        self.assertEqual(out["text"], "Um the endpoint is ready.")

    def test_level_none_disables_backtrack(self):
        out = process("the the report", level="none")
        self.assertEqual(out["text"], "The the report.")

    def test_spoken_punctuation_can_be_disabled(self):
        out = process("I am done period", spoken_punctuation=False, level="none")
        self.assertEqual(out["text"], "I am done period.")

    def test_auto_lists_can_be_disabled(self):
        out = process("Todo one buy the milk two walk the dog", auto_lists=False, level="none")
        self.assertNotIn("\n", out["text"])

    def test_dictionary_and_snippets_both_apply(self):
        out = process(
            "ping clod then sig",
            level="none",
            dictionary=[{"from": "clod", "to": "Claude"}],
            snippets=[{"trigger": "sig", "text": "Best, Sophie"}],
        )
        self.assertEqual(out["text"], "Ping Claude then Best, Sophie.")

    def test_strip_trailing_period(self):
        out = process("hello there", strip_trailing_period=True)
        self.assertEqual(out["text"], "Hello there")

    def test_strip_trailing_period_leaves_question_and_bang(self):
        self.assertEqual(process("is it ready?", strip_trailing_period=True)["text"],
                         "Is it ready?")
        self.assertEqual(process("that is great!", strip_trailing_period=True)["text"],
                         "That is great!")

    def test_lowercase_first(self):
        out = process("Hello there", lowercase_first=True)
        self.assertEqual(out["text"], "hello there.")

    def test_lowercase_first_with_strip_period(self):
        out = process("Hello there", lowercase_first=True, strip_trailing_period=True)
        self.assertEqual(out["text"], "hello there")

    def test_full_pipeline(self):
        out = process(
            "Um, so I was thinking, you know, one finish the report "
            "two send the deck press enter",
            level="high",
        )
        self.assertEqual(out["command"], "enter")
        self.assertEqual(out["text"], "I was thinking:\n1. Finish the report\n2. Send the deck")

    def test_list_formatting_survives_dictionary(self):
        out = process(
            "Goals one ship the api two write the docs",
            level="none",
            dictionary=[{"from": "api", "to": "endpoint"}],
        )
        self.assertEqual(out["text"], "Goals:\n1. Ship the endpoint\n2. Write the docs")

    def test_empty_input(self):
        out = process("")
        self.assertEqual(out["text"], "")
        self.assertFalse(out["changed"])

    def test_whitespace_only_input(self):
        self.assertEqual(process("   \n\t ")["text"], "")

    def test_none_input_is_treated_as_empty(self):
        out = process(None)
        self.assertEqual(out["text"], "")
        self.assertEqual(out["raw"], "")

    def test_single_word(self):
        self.assertEqual(process("hello")["text"], "Hello")

    def test_every_level_runs(self):
        for level in CLEANUP_LEVELS:
            out = process("um, so I like, you know, finished it", level=level)
            self.assertIsInstance(out["text"], str)
            self.assertTrue(out["text"])


class TestPathologicalInputs(unittest.TestCase):
    """Inputs that should never raise, whatever else they do."""

    SAMPLES = [
        "",
        " ",
        "\n\n\n",
        "\t",
        "a",
        ".",
        "...",
        "!!!",
        ",,,",
        "🎉",
        "🎉 🎉 🎉",
        "café naïve résumé",
        "日本語のテキストです",
        "one two three",
        "period comma period",
        "um um um",
        "((()))",
        "C++ vs C#",
        "a" * 5000,
        "word " * 2000,
        "um, " * 500 + "the end",
    ]

    def test_process_never_raises(self):
        for sample in self.SAMPLES:
            for level in CLEANUP_LEVELS:
                with self.subTest(sample=sample[:40], level=level):
                    out = process(
                        sample,
                        level=level,
                        dictionary=[{"from": "C++", "to": "C plus plus"}],
                        snippets=[{"trigger": "sig", "text": "Best"}],
                    )
                    self.assertIsInstance(out["text"], str)

    def test_each_function_never_raises(self):
        for sample in self.SAMPLES:
            with self.subTest(sample=sample[:40]):
                self.assertIsInstance(apply_dictionary(sample, [{"from": "a.b", "to": "x"}]), str)
                self.assertIsInstance(expand_snippets(sample, [{"trigger": "sig", "text": "B"}]), str)
                self.assertIsInstance(apply_spoken_punctuation(sample), str)
                self.assertIsInstance(apply_backtrack(sample), str)
                self.assertIsInstance(format_lists(sample), str)
                self.assertIsInstance(normalize_spacing(sample), str)
                self.assertIsInstance(detect_trailing_command(sample)[0], str)
                for level in CLEANUP_LEVELS:
                    self.assertIsInstance(remove_fillers(sample, level), str)

    def test_long_text_is_handled(self):
        text = ("Um, so I was thinking, you know, we should ship it. " * 200).strip()
        out = process(text, level="high")
        self.assertNotIn("Um,", out["text"])
        self.assertGreater(len(out["text"]), 1000)

    def test_functions_are_pure(self):
        # Same input twice must give the same answer, and the input string
        # (immutable, but the entry list is not) must be unmodified.
        entries = [{"from": "clod", "to": "Claude"}]
        text = "ask clod about the api period Then stop"
        first = process(text, dictionary=entries)
        second = process(text, dictionary=entries)
        self.assertEqual(first, second)
        self.assertEqual(entries, [{"from": "clod", "to": "Claude"}])

    def test_idempotent_on_clean_output(self):
        once = process("um, the report is, like, done period Then we ship")["text"]
        twice = process(once)["text"]
        self.assertEqual(once, twice)


if __name__ == "__main__":
    unittest.main()
