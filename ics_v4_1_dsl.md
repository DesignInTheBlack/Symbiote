# ICS v4.1 DSL (Symbiote Memory Language)

This is the authoritative reference for the ICS v4.1 memory DSL used by Symbiote. It is derived from the live prompt contract in `memory_syntax.md` and the parser/repair implementation in `src-tauri/src/core/memory/dsl.rs`.

This doc describes what the system expects in memory output and what it will reject.

---

## 1) Core Rules

- Memory output must be a single `<memory>` block.
- One statement per line.
- Blank lines and lines starting with `//` are ignored.
- Facts require a **key**. Entity declarations without keys are invalid.
- Relations must use the canonical relation vocabulary or be rejected.
- Modifiers are **end-of-statement tokens only** (confidence, time, scope, source, negation).

---

## 2) Entity References

- `#Label` references or creates an entity by label. Spaces are allowed.
- `$handle` references a special handle. For user and assistant always use `$user` and `$assistant`.
- Do not use `#User` or `#Assistant`.

Examples:
```
#Ken
#Project Alpha
$user
$assistant
```

---

## 3) Facts

Facts store literal values only. They must have a key.

Valid:
```
#Ken:role = "designer"
#Project Alpha.status = "active"
```

Invalid:
```
#Ken = "Ken"        // invalid: missing key
```

---

## 4) Relations

Relations connect entities with roles. Use the canonical relation vocabulary.

Two participants, directed:
```
created_by(creator: $user -> created: $assistant)
```

Two participants, bidirectional:
```
friends(person: #Alice <-> person: #Bob)
```

Multi-participant (comma-separated):
```
collaborates_with(person: #Alice, person: #Bob, person: #Chris)
```

Rules:
- Relation names never use `#`.
- Roles are free-text tokens, but canonical roles improve inference.
- Do not use commas in `->` or `<->` forms.

---

## 5) Modifiers

Modifiers are optional and must appear at the **end** of the statement.

Allowed modifiers:

- Confidence: `~0.0` to `~1.0` or `~NN%`
- Time: `^YYYY-MM-DD`, `^YYYY-MM-DDThh:mm:ssZ`, `^[start..end]`, `^today`, `^yesterday`, `^this_week`
- Scope: `@global`, `@session`, `@project:alpha`, `@context:chat-1`
- Source: `<url-or-id>`
- Negation: `!` or `!deny`

Valid:
```
likes(person: #Bob, object: #Sushi) ~1.0
met(person: $user, person: #Alice) ~0.8 ^yesterday @global
```

Invalid:
```
confidence: 1.00
observed: 17 minutes ago
```

---

## 6) Canonical Relation Vocabulary

Use these exact relation names. Aliases resolve to these.

```
created_by(creator: $user -> created: $assistant)
owns(owner: #Owner -> object: #Object)
works_at(person: #Person -> work: #Work)
works_with(person: #A <-> person: #B)
collaborates_with(person: #A <-> person: #B)
prefers(person: #Person -> object: #Thing)
likes(person: #Person -> object: #Thing)
dislikes(person: #Person -> object: #Thing)
project_member_of(member: #Person -> project: #Project)
member_of(member: #Person -> group: #Group)
lives_in(person: #Person -> place: #Place)
friends(person: #A <-> person: #B)
parent_of(parent: #Parent -> child: #Child)
father_of(father: #Father -> child: #Child)
mother_of(mother: #Mother -> child: #Child)
spouse_of(spouse: #A <-> spouse: #B)
sibling_of(sibling: #A <-> sibling: #B)
employer_of(employer: #Org -> employee: #Person)
writes(subject: #Person -> object: #Work)
```

---

## 7) Role Type Inference (Reference)

These role names influence entity type inference:

- Person: `person`, `user`, `owner`, `author`, `creator`, `subject`, `actor`, `parent`, `child`, `mother`, `father`, `daughter`, `son`, `spouse`, `sibling`, `brother`, `sister`, `partner`, `husband`, `wife`
- Place: `place`, `location`, `city`, `country`, `venue`
- Work: `work`, `project`, `product`, `book`, `movie`, `song`, `company`
- Event: `event`, `meeting`, `appointment`
- Concept: `concept`, `idea`, `topic`, `category`, `object`, `thing`, `item`

---

## 8) Invalid Patterns (Explicitly Rejected)

These will be discarded:

```
#created_by(creator: $user) = "Ken"   // relation name prefixed with #
created = Ergo                         // missing subject
confidence: 1.00                       // invalid modifier format
observed: 17 minutes ago               // invalid modifier format
```

---

## 9) File References

Source of truth in this repo:

- `memory_syntax.md`
- `src-tauri/src/core/memory/dsl.rs`
- `src-tauri/src/core/memory/rel_vocab.rs`
- `src-tauri/src/db/schema.sql`

---

## 10) Minimal Valid Example

```
<memory>
created_by(creator: $user -> created: $assistant) ~0.8
#Project Alpha:status = "active" ^2026-03-01
</memory>
```

