# دليل تنفيذ جزء الـ AI في LegacyLens — خطوة بخطوة

الـ repo ده فيه نسخة شغالة بالفعل من: OCR، كشف الـ PII المصري، موديل أرقام خاص بيك، تصنيف المستندات، الـ Risk Score، الـ Redaction، والـ Policy Suggestions، وكلها شغالة جوه API.
دورك إنك تشغله، تفهم كل جزء، وبعدين تطوره بالمراحل اللي تحت.

---

## المرحلة 0 — جهّز جهازك (ساعة)

1. **Python 3.11**: نزّله من https://www.python.org/downloads/ ، ووانت بتسطب علّم على "Add Python to PATH".
2. **Tesseract OCR**:
   - Windows: نزّل الـ installer من https://github.com/UB-Mannheim/tesseract/wiki . أثناء التسطيب افتح "Additional language data" وعلّم على **Arabic**.
   - بعدها اعمل environment variable اسمه `TESSERACT_CMD` قيمته `C:\Program Files\Tesseract-OCR\tesseract.exe`.
   - Ubuntu: `sudo apt install tesseract-ocr tesseract-ocr-ara`
3. **Git** و **VS Code**.
4. فك الـ zip، وافتح الفولدر في VS Code، وافتح Terminal:

```powershell
python -m venv .venv
.venv\Scripts\activate            # على Linux/Mac:  source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> لو على Linux أو Git Bash، شغّل `bash scripts/download_models.sh` عشان ينزّل الـ Arabic OCR في `models/tessdata`.

---

## المرحلة 1 — شغّل اللي موجود واتأكد إنه شغال (ساعة)

```powershell
$env:PYTHONPATH="src"             # Linux: export PYTHONPATH=src
pytest -q                         # المفروض تطلع 21 passed
uvicorn api.main:app --reload --port 8000
```

افتح http://localhost:8000/docs ، وجرّب `/scan` و`/redact` على صورة من الداتا الـ synthetic (المرحلة 2).
**ماتجربش ببطاقتك الحقيقية** على أي جهاز مش متأمن.

الموديلات الصغيرة (`models/digits_mlp.joblib` و`models/doc_classifier_baseline.joblib`) موجودة جاهزة في الـ zip. لو عايز تعيد تدريبها بنفسك، الأوامر في المرحلة 3.

---

## المرحلة 2 — الداتا: تجيبها منين بالظبط؟

### القاعدة الذهبية
**ممنوع تجمع بطايق أو جوازات سفر حقيقية لناس.** ده مخالف لقانون حماية البيانات الشخصية المصري (151 لسنة 2020)، وكمان بيضرب فكرة المشروع نفسها. الحل إن كل حاجة تبقى **synthetic** وعليها watermark واضح، ومعاها **public datasets** للتقييم.

### (أ) داتا من عندك، synthetic (موجودة في الكود)

| الأمر | بيطلّع إيه |
|---|---|
| `python -m legacylens_ai.data.synthetic_docs --out data/synthetic --n-per-class 300` | 1800 صورة، 6 أنواع: بطاقة، جواز، عقد، كشف حساب، صورة شخصية، ملاحظة. ومعاها `labels.jsonl` فيه كل قيمة PII ومكانها |
| `python -m legacylens_ai.data.synthetic_text --n 3000 --out data/pii_text.jsonl` | 3000 جملة عربي وإنجليزي، فيها PII متعلّم عليه، ومعاها hard negatives زي أرقام طلبات من 14 رقم |

الداتا دي بتطلع تلقائياً بـ train/val/test split. 70% منها بيتعمل لها augmentation (ميل، blur، noise، JPEG، خلفية) كأنها متصورة بموبايل.

**تطوّرها إزاي:** في `src/legacylens_ai/data/synthetic_docs.py` زوّد templates جديدة (توكيل، شهادة ميلاد، فاتورة كهربا)، وفونتات، وأسماء في `fake_values.py`.

### (ب) Public datasets حقيقية (للتقييم والتدريب)

| الداتا | الرابط | هتستخدمها في |
|---|---|---|
| **MIDV-2020** (مستندات هوية mock لدول مختلفة، صور وفيديو) | https://l3i-share.univ-lr.fr/MIDV2020/midv2020.html | تقييم تصنيف id_card/passport على صور حقيقية الشكل |
| **RVL-CDIP** (400 ألف مستند ممسوح، 16 نوع) | https://huggingface.co/datasets/aharley/rvl_cdip | الأنواع other وfinancial (letter, memo, invoice, budget) |
| **CAMeLBERT NER** (موديل جاهز مش داتا) | https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-mix-ner | الأسماء العربي في النص الحر |
| **صور وشوش** | `skimage.data.astronaut()` (صورة NASA public domain)، أو WIDER FACE لو عايز تقييم أكبر | اختبار كشف الوشوش والـ pixelation |

> RVL-CDIP حجمها كبير (حوالي 40GB). الـ notebook بيستخدم **streaming** ويسحب 400 صورة بس لكل نوع، فمش محتاج تنزّلها كلها.

### (ج) Test set حقيقي صغير (اختياري، آخر خطوة)
اطلب من 5 إلى 10 أصحاب **بموافقة مكتوبة** يصوروا بطاقة **وهمية** إنت طابعها (من الـ synthetic generator) بموبايلاتهم، في إضاءة وزوايا مختلفة. كده تختبر "صورة موبايل حقيقية" من غير بيانات حقيقية.

---

## المرحلة 3 — درّب الموديلات بتاعتك (على اللابتوب، CPU)

```powershell
python scripts/train_digit_model.py --strings 3000                       # موديل الأرقام العربي ٠-٩
python -m legacylens_ai.data.synthetic_docs --out data/synthetic --n-per-class 200
python scripts/train_classifier_baseline.py --data data/synthetic        # تصنيف المستندات (baseline)
```

### إزاي تقيّم (ودي الأرقام اللي هتحطها في الـ CV)
```powershell
python scripts/evaluate_pii.py --data data/pii_text.jsonl --out reports/pii_eval.json
python scripts/evaluate_pipeline.py --data data/synthetic --split test --limit 300
```

**النتايج الحالية (synthetic)**:
- الرقم القومي من الصور: 0.78 recall
- الموبايل: 1.00
- الـ IBAN: 0.80
- الجواز: 0.65

**ليه عملنا موديل أرقام خاص؟** Tesseract بيفشل تقريباً بالكامل في قراءة الأرقام الهندية (٠١٢٣...)، ودي الأرقام اللي مطبوعة على البطاقة المصرية. ده الـ "AI بتاعك" اللي تقدر تشرحه في أي interview: لقيت مشكلة، قستها، وحلّيتها بموديل.

---

## المرحلة 4 — موديل CNN على Kaggle (ببلاش، GPU)

1. ارفع الـ repo على GitHub.
2. ادخل https://www.kaggle.com واعمل حساب، ووثّق رقم موبايلك عشان يفتحلك الـ GPU.
3. اعمل **New Notebook**، وبعدين File ← Import Notebook ← اختار `notebooks/kaggle_train_classifier.ipynb`.
4. من Settings: خلّي Accelerator = **GPU T4 x2**، و Internet = **On**.
5. غيّر `<your-user>` في أول خلية باسمك على GitHub، واعمل **Run All** (حوالي 20-30 دقيقة).
6. من تاب Output نزّل `doc_classifier.onnx` و`doc_classifier.json` وحطهم في `models/`.
   الكود بيستخدمهم تلقائياً بدل الـ baseline، لأن `classify/doc_classifier.py` بيدوّر عليهم الأول.

خلية الـ RVL-CDIP في الـ notebook مهمة جداً: بتخلط صور حقيقية مع الـ synthetic. من غيرها، الـ 100% accuracy اللي هتشوفها **وهمية**، لأن الـ templates سهلة.

---

## المرحلة 5 — الأسماء العربي في النص الحر

```powershell
pip install transformers torch
$env:LEGACYLENS_ARABIC_NER="1"
python scripts/evaluate_pii.py --data data/pii_text.jsonl
```

قارن سطر `PERSON_FREE` قبل وبعد التفعيل: كان 0.00، وهتشوف الموديل زوّد قد إيه. ده رقم ممتاز تحطه في الـ README.

---

## المرحلة 6 — الربط مع باقي LegacyLens (الباك إند)

```
User upload ─► Backend ─► POST /scan (الملف في الذاكرة بس) ─► التقرير (قيم masked)
                    │
                    ├─► يعرض للمستخدم: Risk Score + الأسباب + Policy مقترحة (المستخدم يأكد أو يعدّل)
                    └─► يشفّر الملف (per-file key) ويخزنه ─► يخزن التقرير، مش الملف المفتوح
```

قواعد لازم تلتزم بيها:
- الـ AI service يبقى في **network داخلي** ومايتعرضش على الإنترنت.
- مفيش logging للنص أو للصور. الـ report أصلاً بيرجّع قيم masked بس.
- الـ scan يكون **opt-in**: اليوزر يوافق قبل ما الملف يتحلل.
- `/redact` هو اللي بيعمل "Smart Sharing": تشارك نسخة مخفية بدل الأصل.

تشغيل بـ Docker:
```bash
docker build -t legacylens-ai .
docker run -p 8000:8000 legacylens-ai
```

---

## المرحلة 7 — الخطة الزمنية المقترحة

| الأسبوع | الشغل |
|---|---|
| 1 | المرحلة 0-1: شغّل كل حاجة، واقرا كل ملف في `src/` |
| 2 | المرحلة 2-3: كبّر الداتا، وزوّد templates، وطلّع تقارير التقييم |
| 3 | المرحلة 4: CNN على Kaggle + RVL-CDIP، وقيّم على MIDV-2020 |
| 4 | المرحلة 5 + تحسين العناوين والتواريخ (نفس فكرة field re-read) |
| 5 | الربط مع الباك إند + Docker + README فيه الأرقام |

## نقط ضعف معروفة (اشتغل عليها)
1. اسم الأب ساعات بيتقري غلط، فمابيتخفاش في الـ redaction.
2. العنوان وتاريخ الميلاد العربي مش بيتقروا كويس. الحل: field re-read زي الرقم القومي.
3. Haar face detector ضعيف، الأفضل تستبدله بـ YuNet (`cv2.FaceDetectorYN`) أو MediaPipe.
4. الـ digit segmentation بيفشل في حوالي 11% من الصور الـ noisy. الحل: CRNN + CTC على Kaggle.
5. كل الأرقام synthetic. **لازم** تقييم على داتا حقيقية قبل ما تكتب أي accuracy في الـ CV.
