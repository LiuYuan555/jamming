# Lightweight turn-intent classifier

## Decision contract

The runtime target is **intent observable up to the current turn**, not the
evaluator's hidden session scenario.

| Output | Observable definition | Provisional before event? |
|---|---|---|
| `browsing` | The customer is still exploring and has not supplied a concrete clarification | Yes |
| `buying` | The customer has supplied a concrete requirement and no correction/boundary event has occurred | Yes |
| `intent_override` | A correction/retraction is present in the observed history | No |
| `boundary` | The customer explicitly delegates an attribute decision or sets a boundary | No |

This distinction matters. A Boundary first message is byte-for-byte identical
to a Browsing first message in the public simulator. Before the first boundary
reply, no causal classifier can tell them apart. Likewise, a future correction
must not be treated as an observed Intent Override before it happens. The API
therefore marks Buying/Browsing predictions as `provisional` and masks
Boundary/Intent Override until their evidence is observed.

For retrieval routing this contract is safer than pretending to predict the
future. A browsing session may transition to Buying after the customer gives a
specific requirement. Intent Override should trigger removal of the retracted
constraint and a fresh retrieval; Boundary should avoid repeating the refused
question and use a broad fallback retrieval.

## Model

`starter/intent_classifier.py` implements multinomial logistic regression over:

- 384 deterministic hashed word unigram/bigram features from all observed user
  messages, with extra weight for the latest turn;
- conversation length and token counts;
- asked-attribute history;
- observable exploring, requirement, clarification, correction and boundary
  indicators.

The trained prototype has 1,640 parameters and a 34,620-byte JSON artifact. It
requires NumPy but no network, model server, tokenizer download or LLM call.

Runtime integration:

```python
from starter.intent_classifier import IntentClassifier

classifier = IntentClassifier.load("artifacts/intent_classifier.json")
prediction = classifier.predict(
    messages=[
        "I'm looking for Women's Shoes, but I'm still exploring.",
        "For that, what matters is: leather; color: black.",
    ],
    asked_attributes=["other"],
)

route = prediction.label
confidence = prediction.confidence
provisional = prediction.provisional
```

`prediction.probabilities` contains all four class probabilities. Retrieval
should use probability-weighted route blending rather than an abrupt switch
when confidence is low.

## Leakage-safe synthetic training

`tools/train_intent_classifier.py` generates all four scenarios for sampled
catalog products using the public simulator. It applies these safeguards:

1. Public target `parent_asin` values are excluded from training generation.
2. Train/validation/test splitting happens by `parent_asin` before turns are
   expanded, so turns from one product cannot cross splits.
3. Raw identifiers, hidden product fields and hidden scenario are not model
   inputs.
4. Hidden scenario is retained only for diagnostic evaluation.

Example:

```bash
.venv/bin/python tools/train_intent_classifier.py \
  --output /tmp/techjam_intent_classifier.json \
  --dataset-output /tmp/techjam_intent_dataset.jsonl \
  --report /tmp/techjam_intent_train_report.json \
  --products 1000 --max-turns 4 --epochs 180

.venv/bin/python tools/evaluate_intent_classifier.py \
  --model /tmp/techjam_intent_classifier.json \
  --output /tmp/techjam_intent_public_report.json
```

The 1,000-product run produced 16,000 turn examples across 706/143/151
product-disjoint train/validation/test groups. Public target products were not
used for training.

## Empirical result

On 800 turns generated from the held-out public 200 sessions:

| Evaluation target | Accuracy | Macro F1 |
|---|---:|---:|
| Observable turn intent | 1.0000 | 1.0000 |
| Hidden session scenario | 0.6713 | 0.6290 |
| Hidden scenario, first turn only | 0.8000 | 0.4458 |
| Hidden scenario, after Boundary/Override event | 1.0000 | 1.0000 |

Hidden-scenario confusion matrix over all turns (truth rows, prediction
columns; order Buying, Browsing, Intent Override, Boundary):

```text
[[320,   0,  0,  0],
 [164, 156,  0,  0],
 [ 78,   0, 42,  0],
 [  0,  21,  0, 19]]
```

The 100% observable score is a template-matched simulator result, not evidence
of 100% accuracy on natural customer language. The labels are deterministic
from public simulator events, and training/test messages share its grammar.
The useful result is the hidden-scenario gap: 78 pre-event override turns are
correctly treated as Buying, and 21 pre-event boundary turns as Browsing, so
hidden labels score zero on those causally unavailable cases.

Before claiming real-language generalization, add a locked, manually labelled
paraphrase/OOD set. Accuracy, macro F1, calibration and downstream retrieval
score should all be reported.

## External datasets: what maps honestly

No reviewed external dataset provides TechJam's four scenario labels directly.
Naively renaming their dialogue acts would create label noise.

- [U-NEED](https://github.com/LeeeeoLiu/U-NEED) contains 7,698 annotated
  e-commerce pre-sales dialogues with speaker actions, mentioned attributes and
  recommended products. It is the closest source for preference/attribute
  language, but it is Chinese and does not provide these four labels. Use it
  for representation or paraphrase pretraining only, then label a small subset
  with the observable contract above.
- [E-ConvRec](https://aclanthology.org/2022.lrec-1.622/) contains more than
  25,000 real Chinese e-commerce dialogues and defines preference recognition,
  dialogue management and personalized recommendation tasks. Those targets do
  not map to Buying/Browsing/Override/Boundary.
- [SIMMC 2.0](https://aclanthology.org/2021.emnlp-main.401/) has 11,000 shopping
  dialogues and 117,000 utterances. Its simulated-flow plus manual-paraphrase
  collection process is a strong pattern for TechJam augmentation, but its
  multimodal goals and annotations are not the four target intents.
- [MultiWOZ 2.1](https://aclanthology.org/2020.lrec-1.53/) provides more than
  10,000 task-oriented dialogues, corrected dialogue states and user dialogue
  acts. It can support generic slot/state representation, but is neither
  shopping-specific nor labelled for these four intents.

Recommended external-data workflow:

1. Freeze the observable label guide in this document.
2. Sample shopping utterances containing exploration, concrete constraints,
   corrections and explicit delegation/refusal.
3. Have two annotators label complete prefixes, not isolated utterances; resolve
   disagreements.
4. Keep external-source splits separate and report them as OOD evaluation.
5. Use external data for training only after measuring label agreement and
   checking its licence and language/domain shift.

This is more defensible than presenting external pretraining as if its native
labels were equivalent to TechJam scenarios.
