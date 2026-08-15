#!/usr/bin/env python3
"""
DictaFlow text post-processing — the deterministic half of cleanup.

Whisper gives us words; this module gives us *text you can paste*. Everything
here runs in microseconds on the CPU, is pure stdlib, and is a pure function of
its input — no config reads, no clock, no network, no model. That matters for
two reasons:

  1. It runs on every single dictation, in the hot path between "you released
     the key" and "the text appeared". A 200 ms LLM round-trip is a tax you pay
     for real cleanup; you should not also pay it to turn the word "period"
     into a "." or to expand "sig" into your email signature.
  2. It's the only part of the pipeline that can be tested exhaustively. The
     Whisper side is measured with a stopwatch and an ear; this side is
     measured with assertions. tests/test_textproc.py is the spec.

The governing bias throughout: **a missed transformation is invisible, a wrong
one corrupts the sentence.** If a heuristic is ambiguous, it does nothing. Each
place where that bias forced a narrower rule than you'd expect is commented
with the false positive that motivated it.
"""

from __future__ import annotations

import re

# Cleanup aggressiveness. "none" does not mean "do nothing" — see process().
CLEANUP_LEVELS = ("none", "light", "medium", "high")


# ──────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────

# Punctuation we strip off a token before comparing it to a keyword. Whisper
# frequently emits "comma," or "Period." for a *spoken* command, so matching on
# the bare token would miss the very cases we care about most.
_EDGE_PUNCT = ".,!?;:\"'()[]…—-"


def _core(token: str) -> str:
    """Lowercased token with edge punctuation stripped — the comparison key."""
    return token.strip(_EDGE_PUNCT).lower()


def _norm_key(s: str) -> str:
    """Whitespace- and case-insensitive key so "Pull  Request" == "pull request"."""
    return " ".join(s.lower().split())


def _match_case(matched: str, replacement: str) -> str:
    """Carry the *style* of the matched text onto the replacement.

    ALL CAPS -> upper, Capitalized -> capitalize first letter, else verbatim.
    "Verbatim" is the important default: a dictionary entry of {"github":
    "GitHub"} must not be flattened to "Github" just because the speaker's
    lowercase "github" landed at the start of a sentence — the replacement's
    own casing is a deliberate authorial choice.
    """
    letters = [c for c in matched if c.isalpha()]
    if len(letters) > 1 and all(c.isupper() for c in letters):
        return replacement.upper()
    if letters and letters[0].isupper():
        for i, c in enumerate(replacement):
            if c.isalpha():
                return replacement[:i] + c.upper() + replacement[i + 1:]
    return replacement


def _term_pattern(term: str) -> str:
    r"""Whole-word regex for one literal term, metacharacters escaped.

    \b is not usable here. A dictionary entry of "C++" ends in a non-word
    character, so `\bC\+\+\b` never matches — \b requires a word character on
    one side of the position and " " / EOL supply none. Instead we assert
    directly on word characters, which is what "whole word" actually means:
    nothing word-ish may abut the match. That makes "API" refuse to match
    inside "APIs" while "C++" and "a.b" still match at end of line.

    Escaping is not cosmetic: an unescaped "a.b" matches "arb", and an
    unescaped "C++" is a regex syntax error. Every term goes through re.escape.
    """
    words = term.split()
    body = r"\s+".join(re.escape(w) for w in words)
    # Left edge: for a term starting with a word char, forbid a word char
    # before it; for one starting with punctuation, require whitespace/start.
    left = r"(?<!\w)" if (term[0].isalnum() or term[0] == "_") else r"(?<!\S)"
    return left + body + r"(?!\w)"


def _replace_terms(text: str, pairs: list[tuple[str, str]], preserve_case: bool) -> str:
    """Single left-to-right pass replacing any of `pairs`, longest source first.

    One combined alternation rather than N sequential re.sub calls, because
    sequential passes let rule 2 rewrite text that rule 1 just produced —
    {"asap": "as soon as possible"} followed by {"as": "AS"} would corrupt the
    output, and which rules collide would depend on dict ordering. A single
    compiled alternation makes the result order-independent and each character
    of the input eligible for exactly one rule.
    """
    valid = [(f, t) for f, t in pairs if f and f.strip()]
    if not text or not valid:
        return text

    # Longest source first so overlapping rules are deterministic: Python's
    # alternation takes the first branch that matches at a position, not the
    # longest, so ordering *is* the tie-break. sorted() is stable, so equal
    # lengths keep caller order.
    valid = sorted(valid, key=lambda p: -len(p[0].strip()))

    lookup: dict[str, str] = {}
    for src, dst in valid:
        lookup.setdefault(_norm_key(src), dst)

    pattern = "|".join("(?:%s)" % _term_pattern(f.strip()) for f, _ in valid)

    def _sub(m: re.Match) -> str:
        matched = m.group(0)
        dst = lookup.get(_norm_key(matched))
        if dst is None:  # pragma: no cover — only reachable if a key normalizes oddly
            return matched
        return _match_case(matched, dst) if preserve_case else dst

    return re.compile(pattern, re.IGNORECASE).sub(_sub, text)


def _is_abbrev_dot(text: str, i: int) -> bool:
    """True if text[i] is a '.' inside an initialism ("e.g.", "U.S.A.").

    Test: the dot is preceded by a single letter that is not itself preceded by
    an alphanumeric. Used to suppress both "insert a space after the period"
    and "capitalize the next word", either of which mangles "e.g. this" into
    "e. G. this".
    """
    if text[i] != "." or i == 0 or not text[i - 1].isalpha():
        return False
    return i - 2 < 0 or not text[i - 2].isalnum()


def _is_intra_token_dot(text: str, i: int) -> bool:
    """True if text[i] is a '.' inside a single token — a domain, filename,
    version or decimal — rather than the end of a sentence.

    Without this, "arielwalters12@gmail.com" comes out as
    "arielwalters12@gmail. Com": the dot is read as a sentence boundary, a
    space is inserted and the next word is capitalized. Email addresses, file
    names and package versions all show up in dictation constantly (and in
    snippet bodies, which are literal text the user wrote), so corrupting
    them is far worse than the case this gives up.

    The discriminator is the case of the following letter. A real sentence
    boundary is followed by a capital; "gmail.com", "file.py" and "3.14" are
    followed by lowercase or a digit. So "I went home.We left" still splits.
    """
    if text[i] != "." or i == 0 or i + 1 >= len(text):
        return False
    if not text[i - 1].isalnum():
        return False
    nxt = text[i + 1]
    return nxt.isalnum() and not nxt.isupper()


def _tidy(text: str, original: str) -> str:
    """Clean up after a deletion: doubled spaces, orphaned/doubled commas.

    Also restores the leading capital if we deleted a sentence-initial filler —
    removing "So," from "So, the answer is 42" should not leave a lowercase
    sentence for a later stage to maybe-fix.
    """
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]+([,.;:!?])", r"\1", text)
    text = re.sub(r",[\s,]*,", ",", text)          # ", ,"  ", , ,"  -> ","
    text = re.sub(r",\s*([.;:!?])", r"\1", text)   # ", ."          -> "."
    text = re.sub(r"([(\[])\s*,\s*", r"\1", text)  # "( ,"          -> "("
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"^[\s,]+", "", text)
    text = re.sub(r"[ \t]+$", "", text)
    first = next((c for c in original if c.isalpha()), "")
    if first.isupper():
        for i, c in enumerate(text):
            if c.isalpha():
                text = text[:i] + c.upper() + text[i + 1:]
                break
    return text


# ──────────────────────────────────────────────────────────────
# Dictionary + snippets
# ──────────────────────────────────────────────────────────────

def apply_dictionary(text: str, entries: list[dict]) -> str:
    """Apply {"from": ..., "to": ...} vocabulary fixes, case-insensitively.

    This is the "Whisper keeps spelling my coworker's name wrong" fix. Matching
    is whole-word and the replacement inherits the matched text's casing style
    (see _match_case), so one entry covers "api"/"Api"/"API" positions without
    the user writing three.
    """
    if not entries:
        return text
    # Both keys required. A half-written entry {"from": "api"} would otherwise
    # read as {"to": ""} and silently *delete* every occurrence of the word.
    pairs = [(str(e["from"]), str(e["to"])) for e in entries
             if isinstance(e, dict) and "from" in e and "to" in e]
    return _replace_terms(text, pairs, preserve_case=True)


def expand_snippets(text: str, snippets: list[dict]) -> str:
    """Expand {"trigger": ..., "text": ...} macros anywhere in the utterance.

    Unlike apply_dictionary, the expansion is inserted verbatim — a snippet is
    a canned block (an address, a signature, a code fence), and re-casing it
    from the trigger's incidental capitalization would only ever damage it.
    """
    if not snippets:
        return text
    pairs = [(str(s["trigger"]), str(s["text"])) for s in snippets
             if isinstance(s, dict) and "trigger" in s and "text" in s]
    return _replace_terms(text, pairs, preserve_case=False)


# ──────────────────────────────────────────────────────────────
# Spoken punctuation
# ──────────────────────────────────────────────────────────────

# name -> (mark, space_before, space_after)
_PUNCT_COMMANDS: dict[str, tuple[str, bool, bool]] = {
    "period":             (".",    False, True),
    "full stop":          (".",    False, True),
    "comma":              (",",    False, True),
    "question mark":      ("?",    False, True),
    "exclamation mark":   ("!",    False, True),
    "exclamation point":  ("!",    False, True),
    "colon":              (":",    False, True),
    "semicolon":          (";",    False, True),
    # Opening marks invert the "glue it to the previous word" rule: "word(" is
    # never what anyone meant. They hug what follows instead.
    "open paren":         ("(",    True,  False),
    "open parenthesis":   ("(",    True,  False),
    "close paren":        (")",    False, True),
    "close parenthesis":  (")",    False, True),
    "quote":              ('"',    True,  False),
    "open quote":         ('"',    True,  False),
    "unquote":            ('"',    False, True),
    "close quote":        ('"',    False, True),
    "dash":               ("—",    True,  True),   # em dash, spaced both sides
    "hyphen":             ("-",    False, False),  # joins two words: state-of-the-art
    "ellipsis":           ("…",    False, True),
    "new line":           ("\n",   False, False),
    "newline":            ("\n",   False, False),
    "new paragraph":      ("\n\n", False, False),
}

# Commands that open something rather than close it — never emitted at end of
# input, because there is nothing left for them to wrap.
_OPENING_COMMANDS = frozenset({
    "open paren", "open parenthesis", "quote", "open quote",
    "new line", "newline", "new paragraph",
})

# Commands exempt from the "next word starts a new clause" test. Brackets and
# quotes sit *inside* a clause by definition — "the total (before tax) is 40"
# has ordinary sentence continuation on both sides of both marks — so the
# clause-start test is structurally wrong for them. They're safe to relax
# because their names are punctuation jargon nobody utters as prose; the one
# genuinely ambiguous member of the family, bare "quote" ("a famous quote"),
# is deliberately left out and keeps the strict test.
_RELAXED_FOLLOW = frozenset({
    "open paren", "open parenthesis", "close paren", "close parenthesis",
    "open quote", "close quote", "unquote",
    "new line", "newline", "new paragraph",
})

_MAX_COMMAND_WORDS = 2

# If the word *before* the candidate is one of these, the candidate is a noun,
# not a command: "a period of time", "the comma operator", "during that
# period", "she wore a period costume". Determiners and prepositions are the
# tell — you cannot say "the" and then a punctuation mark.
_NOUN_CONTEXT = frozenset("""
a an the this that these those some any each every another other one no
my your his her its our their of in on at for to during over per within
after before by with from into onto about between through above below
short long brief grace waiting transition trial rest quiet dark stone ice
golden rough dry historical entire whole full same next last previous
""".split())

# A punctuation command is only plausible if what follows starts a *new*
# clause. This list is deliberately closed and short; adding open-class words
# to it is how you get "the comma operator" turned into "the, operator".
_CLAUSE_STARTERS = frozenset("""
i we you he she it they and but or so then however therefore also if when
while because although please there this that what who why how now next
finally first second third yes no ok okay maybe actually just don't do does
did can could would should will let's lets i'll i'm we'll we're it's that's
here's there's new
""".split())


def _match_command(tokens: list[str], i: int) -> tuple[str, int] | None:
    """Longest punctuation-command name starting at tokens[i], or None."""
    for n in range(_MAX_COMMAND_WORDS, 0, -1):
        if i + n > len(tokens):
            continue
        name = " ".join(_core(t) for t in tokens[i:i + n])
        if name in _PUNCT_COMMANDS:
            return name, n
    return None


def _is_command_use(tokens: list[str], i: int, name: str, span: int) -> bool:
    """The conservative heuristic. Convert only when ALL of:

      1. A real word precedes the candidate. Nobody opens an utterance with a
         bare mark, and "period costume" as a whole input must stay text.
      2. That preceding word is not a determiner/preposition (_NOUN_CONTEXT) —
         this alone kills "a period of time", "the comma operator", and
         "during that period".
      3. What follows is one of:
           a. end of input (closing marks only — you can end on "." but not "("),
           b. another punctuation command ("close quote period"),
           c. a word that starts a new clause: an explicit clause starter, or a
              capitalized word (Whisper capitalizes after a real sentence break).

    Rule 3 is relaxed for the bracket/quote/newline family (_RELAXED_FOLLOW),
    which lives inside a clause rather than at its edge. Those keep rules 1
    and 2 plus "something must follow".
    """
    if i == 0:
        return False
    prev = _core(tokens[i - 1])
    if not prev or not any(c.isalnum() for c in prev):
        return False
    if prev in _NOUN_CONTEXT:
        return False

    opening = name in _OPENING_COMMANDS
    nxt = i + span
    if nxt >= len(tokens):
        return not opening
    if _match_command(tokens, nxt) is not None:
        return True
    if name in _RELAXED_FOLLOW:
        return True
    if tokens[nxt][:1].isupper():
        return True
    return _core(tokens[nxt]) in _CLAUSE_STARTERS


def _punctuate_line(line: str) -> str:
    tokens = line.split()
    if not tokens:
        return line

    out = ""
    sep = ""  # separator to place before the next chunk
    i = 0
    while i < len(tokens):
        hit = _match_command(tokens, i)
        if hit is not None and _is_command_use(tokens, i, *hit):
            name, span = hit
            mark, space_before, space_after = _PUNCT_COMMANDS[name]
            if space_before and out:
                out += " "
            out += mark
            sep = " " if space_after else ""
            i += span
            continue
        out += (sep if out else "") + tokens[i]
        sep = " "
        i += 1
    return out


def apply_spoken_punctuation(text: str) -> str:
    """Turn spoken punctuation names into marks, but only when unambiguous."""
    if not text or not text.strip():
        return text
    return "\n".join(_punctuate_line(line) for line in text.split("\n"))


# ──────────────────────────────────────────────────────────────
# Backtracking / self-correction
# ──────────────────────────────────────────────────────────────

# Explicit "throw away what I just said" markers. "actually" alone is NOT here
# — it is far more often an intensifier ("I actually think that's right") than
# a correction, so it gets the much narrower short-span rule below.
_BACKTRACK_TRIGGERS = ("scratch that", "no wait", "i mean", "actually no", "correction")

_TRIGGER_RE = re.compile(
    r"(?<!\w)(?:%s)(?!\w)\s*[,.:;!?-]*\s*" % "|".join(
        r"\s+".join(re.escape(w) for w in t.split()) for t in _BACKTRACK_TRIGGERS
    ),
    re.IGNORECASE,
)

# Token classes that can be swapped by a bare "actually". Both sides must be
# the same class for the swap to fire — see apply_backtrack's docstring.
_TIME_RE = re.compile(r"^\d{1,4}(?::\d{2})?(?:\s*)?(?:am|pm|a\.m\.|p\.m\.)?$", re.IGNORECASE)
_WEEKDAYS = frozenset("monday tuesday wednesday thursday friday saturday sunday".split())
_MONTHS = frozenset("""january february march april may june july august september
october november december""".split())


def _swap_class(token: str) -> str | None:
    core = _core(token)
    if not core:
        return None
    if _TIME_RE.match(core) and any(c.isdigit() for c in core):
        return "number"
    if core in _WEEKDAYS:
        return "weekday"
    if core in _MONTHS:
        return "month"
    return None


def _drop_short_span_actually(text: str) -> str:
    """`X actually Y` -> `Y` when X and Y are the same swappable class.

    The documented case is "Let's do coffee at 2 actually 3" -> "...at 3".
    Rejected approach: allowing *any* two single words to swap. That reading is
    structurally identical to "I actually think that's right", which must stay
    intact, so there is no way to have both. Restricting the swap to closed
    classes where a bare replacement is the only sensible reading (numbers and
    times, weekdays, months) keeps the useful case and drops the dangerous one.
    """
    tokens = text.split()
    out: list[str] = []
    i = 0
    while i < len(tokens):
        if (
            _core(tokens[i]) == "actually"
            and out
            and i + 1 < len(tokens)
            and _swap_class(out[-1]) is not None
            and _swap_class(out[-1]) == _swap_class(tokens[i + 1])
        ):
            out.pop()          # drop the superseded value
            i += 1             # and the word "actually"
            continue
        out.append(tokens[i])
        i += 1
    return " ".join(out)


def _collapse_stutters(text: str) -> str:
    """"the the report" -> "the report". Alphabetic words only.

    Digits are excluded on purpose: "1 1" is a legitimate thing to dictate
    (versions, scores, list markers) and collapsing it silently loses data.
    """
    return re.sub(
        r"(?<!\w)([A-Za-z][A-Za-z']*)((?:\s+\1)+)(?!\w)",
        lambda m: m.group(1),
        text,
        flags=re.IGNORECASE,
    )


def apply_backtrack(text: str) -> str:
    """Remove spoken self-corrections.

    Three mechanisms, in order:

      1. Explicit triggers ("scratch that", "I mean", …) delete everything from
         the start of the containing *sentence* through the trigger.
         Comma-scoping was tried and rejected: the canonical utterance is
         "let's meet at 3, scratch that, let's meet at 4", where the trigger
         sits in its own comma clause, so comma-scoping deletes only the
         trigger and dutifully keeps the wrong time. Sentence-scoping is the
         only boundary that does the thing the speaker asked for.
      2. The short-span "actually" swap (see _drop_short_span_actually).
      3. Adjacent identical-word stutters, last, so a deletion that pushes two
         copies of a word together still gets cleaned.

    A trigger with nothing after it is left alone — there is no replacement
    text, so deleting the clause would leave an empty utterance.
    """
    if not text or not text.strip():
        return text

    pos = 0
    while True:
        m = _TRIGGER_RE.search(text, pos)
        if m is None:
            break
        tail = text[m.end():].lstrip(" ,")
        if not tail.strip():
            pos = m.end()
            continue
        head = text[:m.start()]
        boundary = max(head.rfind(c) for c in ".?!\n") + 1
        prefix = head[:boundary]
        text = (prefix + " " if prefix else "") + tail
        pos = len(prefix) + (1 if prefix else 0)

    text = _drop_short_span_actually(text)
    return _collapse_stutters(text)


# ──────────────────────────────────────────────────────────────
# Filler removal
# ──────────────────────────────────────────────────────────────

# Non-words. The repeat quantifiers cover "um"/"umm"/"ummm" without listing
# each. The (?<!\d\s) guard exists because "5 mm" is a measurement, not a
# hesitation — that one cost a corrupted note before it was caught.
_LIGHT_WORDS = r"(?:erm|u+m+|u+h+|hm+|mm+|er)"
# ", uh," — the commas exist only to bracket the hesitation, so both go with
# it: "I think, uh, we should go" -> "I think we should go", not "I think, we
# should go". The trade-off is that a genuinely needed comma next to a filler
# ("After the meeting, um, we can talk") is lost; that reads better than a
# comma left dangling in front of nothing.
_LIGHT_PAREN_RE = re.compile(r",[ \t]*(?<!\d\s)(?<!\w)%s(?!\w)[ \t]*," % _LIGHT_WORDS, re.IGNORECASE)
_LIGHT_RE = re.compile(r"(?<!\d\s)(?<!\w)%s(?!\w)[ \t]*,?" % _LIGHT_WORDS, re.IGNORECASE)

# Removed anywhere they appear as a standalone word.
_PLAIN_FILLERS = ("basically", "literally")

# Removed only when comma-delimited or sentence-initial-with-comma, i.e. when
# the speaker audibly set them off as an aside.
_PARENTHETICAL_FILLERS = ("you know", "i mean", "like")

# Removed unless preceded by a determiner: "some sort of thing" and "what kind
# of car" are meaning-bearing; "it was kind of weird" is not.
_HEDGE_PAIRS = ("sort of", "kind of")
_HEDGE_BLOCKERS = frozenset("a an the this that what which some any each every another one no".split())

_SENTENCE_OPENERS = ("so", "well", "right", "okay", "ok", "alright")


def _strip_parenthetical(text: str, phrase: str) -> str:
    p = r"\s+".join(re.escape(w) for w in phrase.split())
    # Both commas go, not just the phrase between them — see _LIGHT_PAREN_RE.
    text = re.sub(r",[ \t]*%s[ \t]*," % p, "", text, flags=re.IGNORECASE)
    text = re.sub(r"(^[ \t]*|[.!?][ \t]+|\n[ \t]*)%s\s*,\s*" % p, r"\1", text, flags=re.IGNORECASE)
    text = re.sub(r",\s*%s(?=[.!?,]|$)" % p, "", text, flags=re.IGNORECASE)
    return text


def _strip_hedges(text: str) -> str:
    def _sub(m: re.Match) -> str:
        prev = _core(m.group(1) or "")
        return m.group(0) if prev in _HEDGE_BLOCKERS else (m.group(1) or "") + " "

    for phrase in _HEDGE_PAIRS:
        p = r"\s+".join(re.escape(w) for w in phrase.split())
        text = re.sub(r"(\S+)?\s*(?<!\w)%s(?!\w)\s*" % p, _sub, text, flags=re.IGNORECASE)
    return text


def remove_fillers(text: str, level: str) -> str:
    """Strip hesitations and discourse fillers at the requested aggressiveness.

    "like" is the whole reason this function has a heuristic. It is a filler in
    "it was, like, really good" and load-bearing in "I like coffee", "looks
    like rain", "like a boss". Part of speech is not recoverable without a
    parser, so the rule is purely prosodic: **remove "like" only when a comma
    sets it off.** Whisper transcribes the pause around filler-"like" as a
    comma and does not insert one around the real uses, which makes the comma a
    surprisingly reliable proxy — and when it isn't, we keep the word.
    "you know" and "I mean" use the same test.
    """
    if level not in CLEANUP_LEVELS:
        raise ValueError(f"level must be one of {CLEANUP_LEVELS}, got {level!r}")
    if level == "none" or not text or not text.strip():
        return text

    original = text
    text = _LIGHT_PAREN_RE.sub("", text)
    text = _LIGHT_RE.sub("", text)

    if level == "high":
        # Tag questions run before the medium-level "you know" rule, which
        # would otherwise strip ", you know" and leave the "?" behind — turning
        # "We ship Friday, you know?" into the statement-shaped "We ship
        # Friday?". The comma is required: "Is that right?" is a real question
        # and must survive.
        text = re.sub(r"\s*,\s*(?:you know|right)\s*\?\s*$", ".", text, flags=re.IGNORECASE)

    if level in ("medium", "high"):
        for phrase in _PARENTHETICAL_FILLERS:
            text = _strip_parenthetical(text, phrase)
        text = _strip_hedges(text)
        for word in _PLAIN_FILLERS:
            text = re.sub(r"(?<!\w)%s(?!\w)[ \t]*,?" % re.escape(word), "", text, flags=re.IGNORECASE)

    if level == "high":
        openers = "|".join(_SENTENCE_OPENERS)
        text = re.sub(
            # ^[ \t]* not ^: an earlier stage may have left a leading space
            # where a filler used to be, which would otherwise hide the anchor.
            r"(^[ \t]*|[.!?][ \t]+|\n[ \t]*)(?:%s)(?!\w),?\s+" % openers,
            r"\1", text, flags=re.IGNORECASE,
        )

    return _tidy(text, original)


# ──────────────────────────────────────────────────────────────
# Spoken lists
# ──────────────────────────────────────────────────────────────

_ORDINALS: list[tuple[str, ...]] = [
    ("one", "first"), ("two", "second"), ("three", "third"), ("four", "fourth"),
    ("five", "fifth"), ("six", "sixth"), ("seven", "seventh"), ("eight", "eighth"),
    ("nine", "ninth"), ("ten", "tenth"),
]

# If an "item" starts with one of these, the ordinal was a numeral or a noun
# modifier, not a list marker: "one of the two options", "one thing to say",
# "one hour and two minutes".
_NON_ITEM_STARTERS = frozenset("""
of or and but to is are was were am in on at for with from by that which who
whom day days week weeks year years hour hours minute minutes second seconds
month months thing things time times o'clock percent pm more less other
dozen hundred thousand million billion half quarter dollar dollars cent cents
copy copies piece pieces item items page pages person people
""".split())

_MIN_ITEM_WORDS = 2


def format_lists(text: str) -> str:
    """Rewrite a spoken enumeration as a numbered list.

    Requires "one" and "two" (or "first"/"second") as standalone words in
    ascending order, each followed by at least two words that don't look like a
    continuation of a number phrase. Those three conditions together are what
    keep "one of the two options" and "I have one thing to say" as prose — both
    fail on item length or on _NON_ITEM_STARTERS, and they are the reason the
    thresholds are not looser.
    """
    if not text or "\n" in text:
        return text  # already structured; don't re-flow someone's formatting

    spans = list(re.finditer(r"\S+", text))
    tokens = [s.group() for s in spans]
    cores = [_core(t) for t in tokens]

    markers: list[tuple[int, int]] = []
    cursor = 0
    for n, variants in enumerate(_ORDINALS, start=1):
        idx = next((j for j in range(cursor, len(cores)) if cores[j] in variants), None)
        if idx is None:
            break
        markers.append((idx, n))
        cursor = idx + 1

    def _item(k: int) -> list[str]:
        start = markers[k][0] + 1
        end = markers[k + 1][0] if k + 1 < len(markers) else len(tokens)
        return tokens[start:end]

    if len(markers) < 2:
        return text
    items = [_item(k) for k in range(len(markers))]
    # Every marker found must carry a real item. Dropping just the bad trailing
    # marker and folding its words into the previous item was tried first and
    # is wrong: "buy one dozen eggs two loaves of bread three apples" fails
    # only on "three apples", and salvaging the first two turns a shopping
    # sentence into a list. One bad marker means these were numerals all along.
    if any(len(it) < _MIN_ITEM_WORDS or _core(it[0]) in _NON_ITEM_STARTERS for it in items):
        return text

    lead = text[:spans[markers[0][0]].start()].strip().rstrip(",;:")
    lines: list[str] = []
    if lead:
        lines.append(lead + ":")
    for n, item in enumerate(items, start=1):
        body = " ".join(item).strip().strip(",")
        for i, c in enumerate(body):
            if c.isalpha():
                body = body[:i] + c.upper() + body[i + 1:]
                break
        lines.append(f"{n}. {body}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Trailing commands
# ──────────────────────────────────────────────────────────────

_TRAILING_ENTER_RE = re.compile(
    r"(?:(?<=\s)|^)[,\s]*(?:press\s+enter|press\s+return|send\s+it)\s*[.!?,]*\s*$",
    re.IGNORECASE,
)


def detect_trailing_command(text: str) -> tuple[str, str | None]:
    """Split a trailing "press enter" / "send it" off the utterance.

    Only at the very end — "I pressed enter and it sent it to the wrong
    person" contains both phrases mid-sentence and must be left entirely alone.
    """
    if not text or not text.strip():
        return text, None
    m = _TRAILING_ENTER_RE.search(text)
    if m is None:
        return text, None
    return text[:m.start()].rstrip(" ,\t\n"), "enter"


# ──────────────────────────────────────────────────────────────
# Spacing / capitalization
# ──────────────────────────────────────────────────────────────

_NO_SPACE_BEFORE = ".,!?;:)]…"


def normalize_spacing(text: str) -> str:
    """The final pass: spacing, sentence capitals, terminal punctuation."""
    if not text or not text.strip():
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]+([%s])" % re.escape(_NO_SPACE_BEFORE), r"\1", text)
    text = re.sub(r"([(\[])[ \t]+", r"\1", text)
    text = re.sub(r"(?<=[)\]])(?=[A-Za-z])", " ", text)

    # One space after sentence punctuation — skipping initialism dots, which
    # would otherwise turn "e.g." into "e. g." and "U.S.A" into "U. S. A".
    out: list[str] = []
    for i, ch in enumerate(text):
        out.append(ch)
        if ch in ".,!?;:" and i + 1 < len(text) and text[i + 1].isalpha():
            if not (ch == "." and (_is_abbrev_dot(text, i)
                                   or _is_intra_token_dot(text, i))):
                out.append(" ")
    text = "".join(out)

    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return ""

    chars = list(text)
    cap_next = True
    for i, ch in enumerate(chars):
        if ch.isalpha():
            if cap_next:
                chars[i] = ch.upper()
            cap_next = False
        elif ch in "!?\n":
            cap_next = True
        elif ch == ".":
            # Not a sentence end inside "3.14" or "e.g.".
            if not _is_abbrev_dot(text, i) and not _is_intra_token_dot(text, i):
                cap_next = True
    text = "".join(chars)

    # Terminal punctuation only where there's a sentence to terminate. A single
    # dictated word is usually a search term or a variable name, and a
    # multi-line result is a list whose last item doesn't want a period.
    if len(text.split()) >= 2 and "\n" not in text:
        if text[-1] in ",;":
            text = text[:-1] + "."
        elif text[-1].isalnum() or text[-1] in ')"\'…':
            if not text.endswith("…"):
                text += "."
    return text


# ──────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────

def process(
    text: str,
    *,
    level: str = "medium",
    dictionary: list[dict] | None = None,
    snippets: list[dict] | None = None,
    spoken_punctuation: bool = True,
    backtrack: bool = True,
    auto_lists: bool = True,
    strip_trailing_period: bool = False,
    lowercase_first: bool = False,
) -> dict:
    """Run the full pipeline. Returns {text, raw, command, changed}.

    Stage order is load-bearing, not alphabetical:

      1. detect_trailing_command — first, so "press enter" can never be eaten
         by filler removal or turned into a list item.
      2. apply_backtrack — before filler removal, because the two share
         vocabulary: "I mean" is a backtrack trigger AND a medium-level filler.
         Backtrack first means the correction is honoured; the other order
         would delete the trigger and silently keep the retracted text.
      3. remove_fillers.
      4. apply_spoken_punctuation — after filler removal, which can strip the
         comma that was gluing a stray filler to a real clause boundary.
      5. format_lists — after punctuation, so "one" .. "two" are still bare
         words and haven't been rewritten around a spoken comma.
      6. apply_dictionary — after all restructuring, so a replacement can't be
         chopped in half by a later stage inserting a mark inside it.
      7. normalize_spacing.
      8. expand_snippets — dead last, AFTER normalization, because snippet
         bodies are literal text the user wrote. Expanding them earlier let
         normalization rewrite them: an email snippet came out as
         "arielwalters12@gmail. Com", and a multi-line snippet had its blank
         lines collapsed. Running last means a snippet body is inserted
         exactly as authored, formatting and all.
      9/10. strip_trailing_period / lowercase_first — presentation-only tweaks
         for inserting the result mid-sentence, applied after everything else.

    level="none" disables *filler removal and backtracking only*. Dictionary,
    snippets, spoken punctuation, list formatting, the trailing command and
    spacing all still run: those are transcription fidelity, not editorializing,
    and a user who turned cleanup off still expects "period" to become ".".
    """
    if level not in CLEANUP_LEVELS:
        raise ValueError(f"level must be one of {CLEANUP_LEVELS}, got {level!r}")
    raw = text if text is not None else ""
    work = raw

    work, command = detect_trailing_command(work)

    if backtrack and level != "none":
        work = apply_backtrack(work)

    work = remove_fillers(work, level)

    if spoken_punctuation:
        work = apply_spoken_punctuation(work)

    if auto_lists:
        work = format_lists(work)

    work = apply_dictionary(work, dictionary or [])
    work = normalize_spacing(work)
    work = expand_snippets(work, snippets or [])

    if strip_trailing_period and work.endswith(".") and not work.endswith(".."):
        work = work[:-1]
    if lowercase_first and work:
        work = work[0].lower() + work[1:]

    return {"text": work, "raw": raw, "command": command, "changed": work != raw}
