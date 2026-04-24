# Rules of Engagement

Regeln für die Zusammenarbeit in diesem Repo — insbesondere während der
UAI-2026-Rebuttal-Phase. Der Leitgedanke: **erst agil, dann agentisch.** KI
verstärkt, was da ist. Wenn unsere Feedback-Schleifen, Batch-Größen oder
Entscheidungswege schlecht sind, beschleunigt KI uns nicht, sondern vergrößert
nur das Chaos.

## 1. Kleine Batches, kleine PRs

- **Ein PR = eine abgeschlossene, reviewbare Einheit.** Faustregel:
  unter ~300 Zeilen Diff (ohne generierte Files, Plots, Fixtures).
- Wenn ein Issue mehr als einen PR rechtfertigt, **in Subtasks aufteilen**
  (siehe §2). Lieber drei kleine PRs als einen "Feature-Epos-PR".
- Große PRs erzeugen Review-Queues, die exponentiell wachsen. Kognitive
  Überforderung des Reviewers → oberflächliches Review → mehr Bugs. Das ist
  genau der Effekt, den wir bei KI-generiertem Code vermeiden müssen.
- **Boy-Scout-Refactors gehören in eigene PRs**, nicht in Feature-PRs.

## 2. Subtasks innerhalb eines Issues

- Jedes Rebuttal-Issue (#14–#17) wird **vor dem Start** in eine Checkliste
  mit 2–5 Subtasks zerlegt. Jeder Subtask ist ein eigener PR.
- Beispiel für #14 (Deep-Kernel):
  - [ ] `deep_kernel.py` mit Feature-Map + RBF-Basis, ohne Training-Loop
  - [ ] Unit-Test: PSD-Eigenschaft, Gradient-Fluss, Shape-Checks
  - [ ] Integration in bestehende Kernel-API (falls nötig)
- Subtasks werden im Issue als Checklist gepflegt, nicht in losen Kommentaren.
- **Keine Subtask-Inflation.** Wenn ein Issue nur eine Datei und 50 Zeilen
  braucht, ist es ein PR. Subtasks sind ein Werkzeug, kein Ritual.

## 3. Schnelles Feedback

- **Tests laufen lokal in Minuten, nicht Stunden.** Wenn ein Test-Suite
  langsamer wird, ist das ein Problem, das gefixt werden muss, bevor neue
  Features draufkommen.
- Für jeden neuen Kernel / jedes neue Modell: **mindestens ein schneller
  Unit-Test** (PSD, Shape, Gradient), bevor Integrations-Tests oder
  End-to-End-Experimente.
- Lange Experiment-Runs gehören nicht in die Test-Suite — separat triggerbar.
- **Testpyramide:** viele schnelle Unit-Tests, wenige Integrations-Tests,
  nur unverzichtbare End-to-End-Experimente im CI.

## 4. Dezentrale Entscheidungen

- **Im-Team-Entscheidung bevorzugen.** Architektur- oder API-Entscheidungen,
  die lokal zum Issue gehören, werden lokal getroffen — nicht eskaliert.
- Eskalation nur bei echten Cross-Cutting-Themen (breaking changes an
  Public-API, Abhängigkeits-Upgrades, neue Hauptkonzepte).
- Wenn unklar ist, ob eine Entscheidung lokal oder gemeinsam getroffen
  werden soll: im Issue fragen, nicht einfach machen und später diskutieren.
- **Advice-Process** statt Approval-Board: Rat einholen von Betroffenen und
  Fachkundigen, aber die Entscheidung liegt beim Umsetzenden.

## 5. Kontext für jedes Issue

Vor dem ersten Commit zu einem Issue kurz klären:
1. **Warum dieses Issue jetzt?** (Rebuttal-Kritikpunkt, Blocker,
   Dependency?)
2. **Welcher Reviewer wird davon profitieren?** (R1, R2, R3 — oder nur
   internes Housekeeping?)
3. **Was ist der kleinstmögliche Scope**, der dem Rebuttal nützt?
4. **Abhängigkeiten zu anderen Issues?** (#16 braucht #14 + #15 etc.)

Diese vier Fragen kurz im Issue-Kommentar beantworten, bevor wir anfangen.

## 6. Qualität bei KI-Generierung

- **Kein ungeprüfter KI-Code in einen PR.** Jede KI-generierte Änderung
  wird vom menschlichen Autor gelesen, verstanden, und — wenn nötig —
  umgeschrieben.
- Faustregel: *Wenn ich nicht erklären kann, warum eine Zeile so ist,
  wie sie ist, darf sie nicht in den PR.*
- **Theory Building** (Peter Naur): Code, den wir in einem Jahr nicht mehr
  verstehen, ist technische Schuld, auch wenn er heute funktioniert.
- Besonders bei Kernel-Code, Spektralmatrizen, PSD-Garantien — Mathematik
  muss vom Autor durchdrungen sein, nicht nur vom Agent "passend gemacht".

## 7. Rebuttal-spezifische Prioritäten (bis 2. Mai 2026)

- **Kritischer Pfad:** #14 → #15 → #16. #17 (HPO) ist der "Fairness-Booster",
  aber kein Blocker für einen ersten Rebuttal-Result.
- **Jeder gemergte Subtask = sofort verwertbar im Rebuttal-Text**, auch wenn
  das Gesamt-Issue noch nicht fertig ist.
- **Nicht-Code-Antworten** (R1 Δω-Sensitivity, R3 circular-Gaussian-Frage,
  R3 Jitter-Frage) laufen parallel und hängen nicht an den Issues.
- **Camera-Ready-Commitments** sind erlaubt und normal — wir müssen nicht
  alles bis zum 2. Mai gebaut haben.

## 8. Kommunikation

- Updates in Issues, nicht in privaten Chats — damit beide Autoren den
  Stand sehen.
- Wenn etwas blockiert ist (z. B. OpenReview-Zugang, fehlende Daten):
  **sofort** im Issue vermerken, nicht erst nach einem Tag.
- **Keine stillen Scope-Änderungen.** Wenn ein Issue größer wird als
  gedacht, im Issue-Kommentar sagen und entscheiden: aufteilen oder
  scope-reduzieren?
