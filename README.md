# Trading Bot V3 — Technical Documentation (Full)

> ملاحظة منهجية (غير إصلاحيّة): هذا المستند يوثّق **الحالة الحالية** للمشروع اعتماداً على الكود الموجود فعلياً كما تم قراءته. أي عنصر غير موجود ضمن الملفات التي تم فتحها أو عدم الوصول له سيتم توثيقه كـ **Not Found**.

---

## 1. Executive Summary

### ما هو المشروع؟
**Trading Bot V3** هو بوت تداول آلي مبني كـ **Pipeline من عدة طبقات** يجمع أخبار السوق، ويحلّلها بواسطة **DeepSeek**، ويحسب **Sentiment** و**Technical Analysis** و**Multi-Timeframe (H4/H1/M15)**، ثم يستخدم **Voting Engine** لتجميع الإشارات وصولاً إلى **Decision score** و**Direction**. بعد ذلك يمرّ القرار عبر **Risk Engine** (قيود score، ثقة AI، drawdown، خسائر يومية، ارتباط/Correlation، عدد الصفقات المفتوحة). عند اجتياز الفحوصات، يحسب **SL/TP** باستخدام **ATR**، ويحدد **Position Size** بناءً على **Equity** ومسافة SL. أخيراً يفتح الصفقة عبر **QuantDinger API** ويتابع التنفيذ عبر **Reconciliation** و**MT5 Watchdog**.

### الهدف منه
- تقليل قرارات التداول إلى إشارات عالية الجودة عبر دمج:
  - أخبار + AI تحليل
  - sentiment keyword-based
  - مؤشرات فنية من QuantDinger
  - محاذاة متعددة الأطر الزمنية
  - نظام تصويت موزون
  - بوابة ML/XGBoost (p_win) قبل فتح الصفقة
  - قيود مخاطر قبل التنفيذ

### الأزواج المتداولة (Trading Pairs)
في `config.py`:
- `SYMBOLS = ["EURUSD", "XAUUSD", "GBPUSD"]`

### التقنيات المستخدمة
حسب الكود:
- Python
- SQLite (WAL mode)
- DeepSeek API (HTTP)
- Finnhub API (news endpoint)
- RSS feeds عبر `feedparser`
- QuantDinger REST API
- MT5 عبر QuantDinger
- XGBoost (Booster) لبوابة `ML Gate v2`
- Telegram Bot API للتنبيهات والأوامر

### كيف يعمل بشكل عام؟ (Overview)
التدفق العام (high level) داخل `main.py`:
1. Startup: تهيئة DB، تسجيل الدخول لـ QuantDinger، اتصال MT5، بدء Threads (Telegram / MT5 watchdog / Reconciliation).
2. Loop دوري:
   - فحص حالة MT5
   - فحص أخبار عالية التأثير قريباً (pausing / skip)
   - جلب الأخبار وفلترتها
   - لكل زوج:
     - تخصيص الأخبار الخاصة بالرمز
     - DeepSeek AI analysis
     - sentiment analysis
     - Multi-Timeframe technical analysis
     - Voting decision: Direction + final_score + scores/components
     - حساب Confidence
     - حفظ `decisions` في DB
     - تحقق signal valid
     - Risk Engine `can_trade`
     - حساب ATR + SL/TP + position sizing
     - بناء features snapshot للـ ML
     - ML v2 inference (XGBoost)
     - فتح الصفقة عبر QuantDinger
     - تخزين expected snapshot في `execution_dataset`
     - حفظ trade في جدول `trades`
3. Threads:
   - Reconciliation loop (كل 60 ثانية): إغلاق على profit target + مراقبة تعارض الأخبار + مطابقة DB مع QD
   - Feedback loop: تحديث أوزان Voting بناءً على performance
   - Telegram polling + heartbeat

---

## 2. High Level Architecture

### تدفق النظام بالكامل (كما طُلب)

**News**
↓
**DeepSeek**
↓
**Sentiment**
↓
**Technical Analysis**
↓
**MTF Analysis**
↓
**Voting Engine**
↓
**Risk Engine**
↓
**ML Engine**
↓
**Execution**
↓
**Reconciliation**
↓
**Training Loop**

> ملاحظة: جزء “Training Loop” لا يتم داخل `main.py` مباشرةً؛ يوجد orchestrator يومي لتدريب/إعادة تحميل نموذج XGBoost على أساس DB. الملف المسؤول: `analysis/models/system_orchestrator.py`.

### دور كل مرحلة

#### 1) News
- مصدر الأخبار: RSS feeds + Finnhub.
- يتم بناء `NewsItem` مع:
  - `source_weight` و`decay` و`is_high_impact`
- ثم يتم فرز/تصفية الأخبار عبر decay وweights.

المكوّنات:
- `data/news/fetcher.py` (fetch_rss_news, fetch_finnhub_news, filter_news_for_symbol)
- `data/news/scoring.py` (score_news_item / filter_relevant_news) — **يظهر استدعاءه في main.py**.

#### 2) DeepSeek
- `analysis/ai/deepseek.py` يأخذ headlines مرتبطة بالرمز.
- يرجع JSON يحتوي:
  - `impact_score` (0-100)
  - `bias` (bullish/bearish/neutral)
  - `confidence` (0-1)
  - `reason` + lists (`key_factors`, `risk_factors`)

#### 3) Sentiment
- sentiment keyword-based:
  - `analysis/sentiment/analyzer.py`
- يحسب `SentimentData(score, direction, bullish_count, bearish_count)`.

#### 4) Technical Analysis
- منطق فني من QuantDinger (proxy مؤشرات):
  - `analysis/technical/indicators.py`
- يوفّر:
  - `get_trend_score` → (score 0-100, direction)
  - `get_momentum_score` → (score 0-100, direction)
  - `get_volatility_score` → float 0-100

#### 5) MTF Analysis
- `analysis/multi_timeframe/analyzer.py`:
  - H4 = trend
  - H1 = decision/momentum
  - M15 = timing short-term momentum
- يقرر:
  - `aligned` (True/False)
  - `strength` (strong/moderate/weak)

#### 6) Voting Engine
- `decision/voting_engine.py`
- يجمع votes من:
  - AI bias + (ai_confidence multiplier)
  - Trend direction
  - Momentum direction
  - Sentiment score thresholds
- يحسب:
  - `final_score` (محدود إلى 100)
  - `direction` = BUY/SELL/NEUTRAL
- يعدّل final_score بناءً على MTF alignment.

#### 7) Risk Engine
- `risk/risk_engine.py`
- `can_trade(...)` ينفذ تحقق تسلسلي:
  1. final_score >= MIN_SCORE
  2. ai_confidence >= AI_MIN_CONFIDENCE
  3. daily loss limit (3% threshold تقريبياً)
  4. drawdown tiers عبر `check_drawdown` 
  5. Max open trades
  6. consecutive losses >= STOP_AFTER_LOSSES
  7. correlation filter عبر `CORRELATION_GROUPS`
  8. duplicate check: symbol/direction مفتوح

#### 8) ML Engine
- ML Gate v2 داخل `main.py` عبر:
  - `analysis/models/xgboost_v2_inference.py`
- يعتمد على تحميل `models/xgb_model.json` ثم:
  - يبني feature vector (10 features كما هو في الكود)
  - ينفذ `model.predict(DMatrix)`
  - ينتج:
    - `p_win`
    - `available`
- Acceptance:
  - إذا not available → bypass
  - else إذا p_win < 0.60 → reject
  - إذا available → size multiplier عبر `get_size_multiplier(p_win)`

> يوجد أيضاً pipeline training منفصل (Section 11 و 12) عبر builder وtrainer.

#### 9) Execution
- فتح الصفقة يتم عبر `execution/quantdinger_client.py`:
  - `open_trade(symbol, direction, size, sl, tp, reason)`
- يمر عبر API endpoint:
  - POST `{QUANTDINGER_URL}/api/mt5/order`
- عند نجاح الطلب يرجع قاموس يتضمن:
  - `status`, `order_id`, `price` ...

#### 10) Reconciliation
- `execution/reconciliation.py`
- Loop كل 60 ثانية:
  - `check_profit_targets(qd_positions)`:
    - يطبق Profit target = 50 USD + trailing trigger/lock
    - عند الإغلاق: `close_trade(order_id)` ثم `close_trade_db_by_order_id(order_id, pnl)`
    - ثم `upsert_execution_actual(...)` لكتابة actual facts
    - كذلك notify telegram
  - `check_news_conflict(qd_positions)`:
    - إذا اتجاه الصفقة عكس bias AI مع `confidence >= 0.75` → إغلاق
  - `reconcile()`:
    - DB trade موجود لكن ليس في QD → close in DB
    - QD orphan positions → warning

#### 11) Training Loop (Closed loop learning / retrain)
- يوجد orchestrator يومي في:
  - `analysis/models/system_orchestrator.py`
- يقوم:
  - يقرأ إحصاءات من `execution_dataset`
  - يقرر should_retrain
  - ينفذ `train_model_from_db(strict_mode=True)`
  - ثم يقوم `load_latest_model(force_reload=True)`

---

## 3. Project Structure

### شجرة المشروع كاملة (وفق ما تم رصده/قراءته)

```text
trading_bot_v3/
├── analysis/
│   ├── ai/
│   │   └── deepseek.py
│   ├── features/
│   │   ├── feature_builder.py
│   │   └── ml_dataset_builder.py
│   ├── multi_timeframe/
│   │   └── analyzer.py
│   ├── models/
│   │   ├── drift_detector.py
│   │   ├── model_manager.py
│   │   ├── performance_monitor.py
│   │   ├── xgboost_trainer.py
│   │   ├── xgboost_inference.py
│   │   ├── xgboost_v2_inference.py
│   │   └── system_orchestrator.py
│   ├── sentiment/
│   │   └── analyzer.py
│   └── technical/
│       ├── indicators.py
│       └── regime.py
├── core/
│   ├── models.py
│   └── exceptions.py
├── data/
│   ├── news/
│   │   ├── fetcher.py
│   │   └── scoring.py
│   ├── market/
│   │   ├── client.py
│   │   └── hybrid_client.py
│   └── storage/
│       └── database.py
├── decision/
│   ├── confidence_engine.py
│   ├── signal_engine.py
│   └── voting_engine.py
├── execution/
│   ├── quantdinger_client.py
│   ├── reconciliation.py
│   └── mt5_watchdog.py
├── feedback/
│   ├── adaptive_weights.py
│   ├── learning.py
│   └── performance.py
├── reports/
│   └── (Not Found)
├── scheduler/
│   └── (Not Found)
├── telegram/
│   ├── notifier.py
│   └── telegram_bot.py
├── utils/
│   └── (Not Found)
├── COMMANDS.py
├── IMPLEMENTATION_COMPLETE.md
├── STARTUP(bat)
├── TODO.md
├── scripts/ (Not Found)
├── main.py
├── config.py
├── train_pipeline.py
├── startup.bat
└── requirements.txt
```

> ملاحظة توثيقية: بعض الملفات المشار إليها في `main.py` لم يتم فتحها ضمن هذه الجولة (مثل `analysis/technical/regime.py` تم فتحها، لكن `analysis/models/performance_monitor.py` و`core/models.py` و`core/exceptions.py` و`utils/logger.py` وملفات أخرى لم تُقرأ نصياً هنا). عند عدم توفر قراءة مباشرة سيتم توثيقها كـ **Not Found** في توثيق الملف-ب-الملف.

### وظيفة كل مجلد
- `analysis/`: تحليل الأخبار والـ ML features وMulti-Timeframe وTechnical indicators وmodels الخاصة بـ XGBoost.
- `data/`: جلب الأخبار وتحويلها إلى NewsItem + تخزين SQLite + (Market clients) لقراءة مؤشرات/شموع.
- `decision/`: منطق Signal وVoting لتوليد `direction` و`final_score`.
- `risk/`: تحقق مخاطر قبل التنفيذ + SL/TP + position sizing.
- `execution/`: فتح الصفقة عبر QuantDinger وReconciliation ومراقبة MT5.
- `feedback/`: تحديث أوزان Voting بناء على performance + (learning hooks) + metrics.
- `telegram/`: إرسال إشعارات وتنفيذ أوامر.

---

## 4. File-by-File Documentation

> فقط الملفات التي تم فتح محتواها سيتم توثيقها بدقة (مع تفاصيل الدوال/المدخلات/المخرجات). أما غير المقروءة هنا فستكون **Not Found**.

### 4.1 `main.py`
**المسؤولية:**
- نقطة الدخول الرئيسية.
- تنسيق كل الطبقات: News → AI → Sentiment → Technical → MTF → Voting → Risk → ML Gate → Execution → DB writes.
- بدء Threads: Telegram + MT5 watchdog + Reconciliation + feedback loop.

**الدوال الموجودة:**
- `run_cycle()`:
  - فحص MT5 status
  - فحص أخبار عالية التأثير قريبة (is_high_impact_soon)
  - fetch news
  - لكل symbol:
    - filter_news_for_symbol
    - analyze_news (DeepSeek)
    - generate_signal (signal_engine)
    - analyze_sentiment (sentiment analyzer)
    - get_multi_timeframe_analysis
    - get_trend_score / get_momentum_score / get_volatility_score / get_market_regime
    - make_decision (voting_engine)
    - calculate_confidence (confidence_engine — Not Found content in this session)
    - save_decision (DB)
    - skip if direction NEUTRAL or signal invalid
    - can_trade (risk_engine)
    - get ATR + entry_price
    - calculate_sl_tp + calculate_position_size
    - build_trade_features → features dict
    - predict_with_v2 → v2_result
    - should_trade_v2 + get_size_multiplier
    - open_trade
    - upsert_execution_expected(order_id, expected_*)
    - save_trade
    - notify_trade_opened
  - `run_feedback_loop()` بعد حلقة الرموز

- `main()`:
  - init_db()
  - login() + set_market_token()
  - connect_mt5()
  - start_telegram_bot(bot_state)
  - start_mt5_watchdog()
  - start_reconciliation()
  - loop forever: run_cycle + sleep NEWS_CHECK_INTERVAL

**المدخلات (Inputs):**
- يعتمد على `config.py` وبيانات البيئة (DEEPSEEK_API_KEY/FINNHUB...)
- اتصال شبكي لـ DeepSeek وFinnhub وQuantDinger

**المخرجات (Outputs):**
- كتابة إلى SQLite:
  - `decisions`, `trades`, `execution_dataset` (expected)
- فتح صفقات عبر QuantDinger
- إشعارات Telegram عبر `telegram/notifier.py`

**من يستدعيه؟**
- يتم تشغيله مباشرة عبر `python main.py`.

**ما الملفات التي يعتمد عليها؟**
- `config.py`
- DB layer: `data/storage/database.py`
- News: `data/news/fetcher.py`, `data/news/scoring.py`
- Market: `data/market/hybrid_client` و`data/market/client`
- Analysis: `analysis/ai/deepseek.py`, `analysis/sentiment/analyzer.py`, `analysis/technical/indicators.py`, `analysis/technical/regime.py`, `analysis/multi_timeframe/analyzer.py`
- Decision: `decision/voting_engine.py`, `decision/signal_engine.py`, `decision/confidence_engine.py` (**Not Found content**)
- Risk: `risk/risk_engine.py`, `risk/sltp.py`, `risk/position_sizing.py`
- ML v2: `analysis/models/xgboost_v2_inference.py`
- Execution: `execution/quantdinger_client.py`, `execution/reconciliation.py`, `execution/mt5_watchdog.py`
- Feedback: `feedback/adaptive_weights.py`
- Telegram: `telegram/telegram_bot.py`, `telegram/notifier.py`

---

### 4.2 `config.py`
**المسؤولية:**
- تخزين إعدادات مفاتيح APIs وMT5 وTrading pairs وRisk thresholds وweights.

**المدخلات:**
- environment variables عبر `dotenv.load_dotenv()`.

**المخرجات:**
- ثوابت Python تُستورد في باقي الملفات.

**أهم المتغيرات (كما ظهر في المقروء):**
- DeepSeek/ Finnhub / Telegram keys
- QuantDinger URL/username/password
- MT5 credentials
- `SYMBOLS`
- Timeframes: `TF_TREND=H4`, `TF_DECISION=H1`, `TF_TIMING=M15`
- Risk: `BASE_RISK_PERCENT`, `MAX_DAILY_LOSS`, `MAX_DRAWDOWN_HALT`, ...
- Voting weights: `INITIAL_WEIGHTS`, `MIN_SCORE`, `AI_MIN_CONFIDENCE`
- News: `NEWS_SOURCE_WEIGHTS`, `NEWS_DECAY_HOURS`, `NEWS_CHECK_INTERVAL`
- Execution retries/timeouts
- Feedback learning params
- Correlation: `MAX_CORRELATION`, `CORRELATION_LOOKBACK` (**ملاحظة: MAX_CORRELATION وLOOKBACK لا يظهر استخدام مباشر في risk_engine المقروء— risk_engine يعتمد على correlation groups فقط**)

---

### 4.3 `data/news/fetcher.py`
**المسؤولية:**
- جلب أخبار من:
  - RSS feeds
  - Finnhub news endpoint
- فلترة حسب كلمات مفتاحية عامة/خاصة بالرمز.
- حساب decay وis_high_impact.

**دوال/وظائف:**
- `fetch_rss_news() -> List[NewsItem]`
  - يستخدم `feedparser.parse`
  - يستبعد headline المكرر
  - يستبعد إذا لم يحتوي على كلمات Forex
  - يستبعد الأخبار القديمة (>8 ساعات)
  - يحسب decay: `calculate_decay`
  - يحدد is_high_impact إذا وجدت keywords HIGH_IMPACT_KEYWORDS
  - يضيف NewsItem لكل entry
  - ثم يجلب Finnhub ويضيف النتائج
  - يفرز القائمة عبر `source_weight * decay`.

- `fetch_finnhub_news() -> List[NewsItem]`
  - يستدعي `https://finnhub.io/api/v1/news?category=forex&token=...`
  - يفلتر ويحدد decay بنفس المنطق.

- `filter_news_for_symbol(news, symbol) -> List[NewsItem]`
  - يفلتر بناءً على `SYMBOL_KEYWORDS[symbol]`.
  - إذا لم يوجد match → يرجع القائمة الأصلية.

**يعتمد على:**
- `core.models.NewsItem` (**Not Found content for models definition**)
- `config.py` (FINNHUB_API_KEY, NEWS_SOURCE_WEIGHTS, NEWS_DECAY_HOURS)

**مخرجات:**
- قائمة NewsItem جاهزة لاستهلاك DeepSeek.

---

### 4.4 `data/news/scoring.py`
**المسؤولية:**
- تحويل NewsItem إلى درجة relevance.

**الدوال:**
- `score_news_item(item: NewsItem) -> float`
  - base_score = source_weight * decay
  - boost 1.5 إذا `is_high_impact`

- `sort_news_by_relevance(news_list) -> List[NewsItem]`
- `filter_relevant_news(news_list, top_n=20) -> List[NewsItem]`
  - تحافظ دائماً على high_impact ثم الباقي.

---

### 4.5 `analysis/ai/deepseek.py`
**المسؤولية:**
- إرسال headlines إلى DeepSeek وتحويل رد JSON إلى `AINewsAnalysis`.

**الدالة الرئيسية:**
- `analyze_news(news_list: List[NewsItem], symbol: str) -> AINewsAnalysis`

**المدخلات:**
- قائمة NewsItem + symbol

**المخرجات (AINewsAnalysis):**
- impact_score: 0-100
- news_impact_score = impact_score (direct transform)
- bias: bullish/bearish/neutral
- confidence: 0-1
- reason: نص
- key_factors / risk_factors

**التعامل مع الأخطاء:**
- عند Exception يرجع neutral باحتمالات 0.

**يعتمد على:**
- `config.DEEPSEEK_API_KEY`
- `core.models.NewsItem` و `core.models.AINewsAnalysis` (**Not Found content**)

---

### 4.6 `analysis/sentiment/analyzer.py`
**المسؤولية:**
- sentiment keyword matching.

**الدالة:**
- `analyze_sentiment(news_list: List[NewsItem], symbol: str) -> SentimentData`

**المنطق:**
- يحدد keywords خاصة بالرمز.
- يحسب bullish_count / bearish_count عبر BULLISH_KEYWORDS وBEARISH_KEYWORDS.
- النتيجة:
  - إذا لا توجد relevant_news → score=50 neutral
  - bullish_ratio > 0.6 → bullish
  - bullish_ratio < 0.4 → bearish
  - else neutral

---

### 4.7 `analysis/technical/indicators.py`
**المسؤولية:**
- استخراج score/direction من مؤشرات QuantDinger (عبر hybrid client).

**الدوال:**
- `get_trend_score(symbol) -> (score: float, direction: str)`
  - uses get_indicators_hybrid(symbol, TF_TREND)
  - إذا ma_trend يحتوي نصوص:
    - "strong uptrend" → (85, bullish)
    - "uptrend" → (70, bullish)
    - "strong downtrend" → (85, bearish)
    - "downtrend" → (70, bearish)
  - fallback على RSI thresholds

- `get_momentum_score(symbol, timeframe=TF_DECISION) -> (score, direction)`
  - RSI-based thresholds

- `get_volatility_score(symbol) -> float`
  - returns (lower = high volatility):
    - “very high”→20
    - “high”→35
    - “low”→80
    - default 55

---

### 4.8 `analysis/technical/regime.py`
**المسؤولية:**
- تحديد regime string بناءً على trend/volatility.

**الدالة:**
- `get_market_regime(symbol) -> str`
  - إذا vol_score < 30 → "HIGH_VOLATILITY"
  - else إذا trend_dir != "neutral" → "TRENDING"
  - else إذا vol_score > 70 → "LOW_VOLATILITY"
  - else → "RANGING"

- `is_safe_to_trade(symbol) -> bool`
  - يرجع True إذا regime ليس من HIGH_VOLATILITY/UNKNOWN.

---

### 4.9 `analysis/multi_timeframe/analyzer.py`
**المسؤولية:**
- تحليل multi-timeframe ومحاذاة الإشارات.

**الدالة:**
- `get_multi_timeframe_analysis(symbol) -> MultiTimeframeData`

**المنطق:**
- H4: get_trend_score
- H1: get_momentum_score
- M15: get_momentum_score(symbol, timeframe=TF_TIMING)
- alignment:
  - strong: 3 non-neutral متطابقة
  - moderate: 2+ non-neutral متطابقة أو H4==H1
  - weak: H1==M15 أو else.

**مخرجات:**
- MultiTimeframeData instance عبر `core.models.MultiTimeframeData` (**Not Found content**)

**إضافة logs:**
- Verification logs عبر get_candles/get_indicators_hybrid (tf_bars + indicators).

---

### 4.10 `decision/signal_engine.py`
**المسؤولية:**
- تحويل AI bias إلى direction وإنتاج signal validity.

**الدوال:**
- `bias_to_direction(bias) -> str`
  - bullish→BUY, bearish→SELL else NEUTRAL

- `get_signal_strength(score, confidence) -> str`
  - مستويات 4: قوي جداً/قوي/متوسط/ضعيف

- `generate_signal(ai_analysis: AINewsAnalysis) -> dict`
  - is_valid:
    - confidence >=0.60
    - bias in [bullish,bearish]
    - score >=55

---

### 4.11 `decision/voting_engine.py`
**المسؤولية:**
- تجميع votes موزونة لإنتاج final_score و direction.

**الدالة:**
- `make_decision(symbol, ai_analysis, trend_data, momentum_data, volatility_score, sentiment_score_val, mtf_data) -> dict`

**المكونات: **
- weights = get_weights(symbol) (DB)
- votes bullish/bearish:
  - AI: multiplier ai_vote_weight = 1.5 إذا ai_confidence >=0.85 else 1.0
  - Trend: +1.0
  - Momentum: +0.75
  - Sentiment score thresholds:
    - >=70 → bullish +0.5
    - <=30 → bearish +0.5

**حساب direction:**
- bullish votes > bearish votes → BUY
- bearish > bullish → SELL
- equal → NEUTRAL

**حساب final_score:**
- ai_weight_boosted = weights['ai'] * (1 + ai_confidence*0.3)
- final_score = (ai_score*ai_weight_boosted + trend_score*weights['trend'] + momentum_score*weights['momentum'] + sentiment_score*weights['sentiment'] + volatility_score*weights['volatility']) * dir_multiplier

**تعديل MTF:**
- إذا not mtf_aligned:
  - strength strong → reduce 25%
  - else → reduce 10%
- إذا aligned and strength strong → boost 10%

---

### 4.12 `risk/risk_engine.py`
**المسؤولية:**
- منظومة فحص قبل فتح الصفقة.

**الدوال:**
- `check_correlation(symbol, direction) -> (bool, reason)`
  - groups:
    - USD_SHORT: EURUSD, GBPUSD, XAUUSD
    - USD_LONG: USDJPY, USDCAD, USDCHF
  - إذا new_group unknown → allow
  - إذا يوجد 2+ open trades لنفس direction داخل نفس group → block

- `can_trade(symbol, direction, final_score, ai_confidence, equity) -> (ok: bool, reason: str)`
  - score check
  - AI confidence check
  - daily loss check (تقريب): starting_balance = equity - stats.total_pnl
    - إذا stats.total_pnl < 0 و daily_loss_pct >= 0.03 → reject
  - drawdown check عبر `risk/drawdown.py`
  - max open trades
  - consecutive_losses >= STOP_AFTER_LOSSES
  - correlation filter
  - duplicate check via `is_symbol_open(symbol, direction)`

---

### 4.13 `risk/drawdown.py`
**المسؤولية:**
- tiers لحماية الحساب.

**الدالة:**
- `check_drawdown(daily_pnl: float, equity: float) -> dict`

**الإجراءات:**
- daily_dd = abs(daily_pnl)/equity إذا daily_pnl <0 else 0
- إذا daily_dd >= MAX_DRAWDOWN_HALT → action halt_day
- إذا daily_dd >= ACCOUNT_DRAWDOWN_STOP → full_stop
- إذا daily_dd >= ACCOUNT_DRAWDOWN_HALF → half_risk
- else ok

> ملاحظة: يتم استخدام `daily_pnl` كـ stats.total_pnl من daily stats.

---

### 4.14 `risk/position_sizing.py`
**المسؤولية:**
- تحديد حجم الصفقة اعتماداً على equity ومسافة SL.

**الدالة:**
- `calculate_position_size(equity, sl_distance, symbol, consecutive_losses=0, score=65.0) -> float`

**المنطق:**
- risk_percent يتغير بناءً على score (>=90..else)
- يتم تخفيض risk_percent بعد consecutive_losses (>=1/2/3)
- يتم تخفيض risk_percent بناءً على equity thresholds (<95000 etc)
- risk_amount = equity * risk_percent
- size = risk_amount / sl_distance
- تطبيق سقوف:
  - MAX_LOT_PER_SYMBOL محدد
  - round size إلى منزلتين
  - ensure >= MIN_LOT

**دالة إضافية:**
- `get_dynamic_risk_multiplier()` (تُظهر في ملف لكن **غير مستخدمة في main.py المقروءة**؛ يعتمد main على calculate_position_size فقط + ML multiplier)

---

### 4.15 `risk/sltp.py`
**المسؤولية:**
- حساب SL/TP باستخدام ATR مع cap.

**الدالة:**
- `calculate_sl_tp(symbol, entry_price, direction, atr) -> (sl, tp)`

**المنطق:**
- pip_value = PIP_VALUES[symbol]
- max_sl_distance = MAX_SL_PIPS * pip
- sl_distance = min(atr*ATR_SL_MULTIPLIER, max_sl_distance)
- tp_distance = atr*ATR_TP_MULTIPLIER
- direction BUY/SELL لتحديد إشارة sl/tp

---

### 4.16 `execution/quantdinger_client.py`
**المسؤولية:**
- عميل API للاتصال بـ QuantDinger.
- فتح وإغلاق صفقات.
- جلب equity ومواقع open positions.

**ملاحظة حرجة (موثقة فقط):**
- يوجد `login()` معرف مرتين داخل نفس الملف. هذا قد يؤدي لتظليل الدالة الأولى (Python override) لكنه ضمن “to-document-current-state”.

**الدوال الموجودة (كما ظهرت):**
- `login() -> str`
- `get_token() -> str`
- `get_headers() -> dict`
- `open_trade(symbol, direction, size, sl, tp, reason) -> dict`
- `close_trade(trade_id) -> bool`
- `get_open_positions() -> list`
- `get_equity() -> float`
- `connect_mt5() -> bool`
- `check_mt5_status() -> bool`

**المخرجات:**
- open_trade يرجع dict status success/error.

---

### 4.17 `execution/reconciliation.py`
**المسؤولية:**
- حلقة reconciliation لمطابقة الحالة.
- إغلاق صفقات على profit target + trailing stop.
- مراقبة تعارض الأخبار.

**الدوال الرئيسية:**
- `check_profit_targets(qd_positions)`
  - لكل pos في qd_positions:
    - order_id = pos.id أو pos.ticket
    - profit = pos.profit
    - direction_norm = normalize type
    - إذا profit >= PROFIT_TARGET_USD → close_trade(order_id) ثم:
      - upsert_execution_actual(..., execution_quality_score=...)
      - close_trade_db_by_order_id(order_id, pnl=profit)
      - notify_trade_closed

- `check_news_conflict(qd_positions)`
  - كل NEWS_MONITOR_INTERVAL: يجلب news ثم analyze_news لكل pos.
  - إذا تعارض direction_norm مع ai.bias وبشرط ai.confidence >= MIN_CONFIDENCE_TO_EXIT → close.

- `reconcile() -> dict`
  - db_trades = get_open_trades()
  - qd_positions = get_open_positions()
  - يقفل DB orphan trades (order_id ليس ضمن qd_ids)
  - log mismatches

- `start_reconciliation(interval=60)`
  - يبدأ thread daemon

**ملاحظة توثيقية:**
- `_record_live_performance` يستخدم `expected_ai_confidence` كproxy لـ expected_p_win.

---

### 4.18 `execution/mt5_watchdog.py`
**المسؤولية:**
- مراقبة الاتصال بـ MT5 عبر QuantDinger
- إعادة الاتصال عند انقطاع.

**الدوال:**
- `check_mt5_connection() -> bool`
- `reconnect_mt5() -> bool`
  - يضمن refresh token عبر `_login()` ثم يفحص حساب ثم connect
- `watchdog_loop()`
- `start_mt5_watchdog()`

---

### 4.19 `analysis/models/system_orchestrator.py`
**المسؤولية:**
- orchestrator يومي لإعادة تدريب نموذج XGBoost بناءً على DB.

**الدوال:**
- `_get_execution_dataset_stats()`
  - يقرأ COUNT(*) و MAX(dataset_updated_at)
  - new_rows_count = total (approx)

- `run_daily_cycle()`
  - إذا should_retrain(new_rows_count, last_train_ts) → train_model_from_db(strict_mode=True)
  - ثم `load_latest_model(force_reload=True)`

- `daily_thread_runner(hour=0, minute=5, interval_sec=60)`
- `start_daily_orchestrator_thread(hour=0, minute=5)`

---

### 4.20 `analysis/features/feature_builder.py`
**المسؤولية:**
- بناء feature snapshot للـ ML ويطابق expected_* columns في `execution_dataset`.

**الدالة:**
- `build_trade_features(symbol, market_data, indicators, ai_analysis, sentiment, regime, mtf_data) -> dict`

**الخصائص التي يعيدها (expected structure):**
- rsi, atr, macd (normalized)
- trend_strength, momentum_score, volatility_score
- market_regime (signed)
- session (normalized from UTC mapping)
- spread (guarded)
- ai_score, sentiment_score, news_impact_score
- expected_entry, expected_confidence, expected_final_score, direction (signed)

**منطق مهم:**
- يحاول تجنب None عبر fallbacks، ويضمن قيم افتراضية لتلافي NULL.

---

### 4.21 `analysis/features/ml_dataset_builder.py`
**المسؤولية:**
- تحويل صف `execution_dataset` إلى (X, y) جاهزة لتدريب ML.

**الثوابت:**
- `STRICT_MODE = True`

**الدوال:**
- `build_ml_row(execution_row) -> Optional[Tuple[List[float], float]]`
  - يستخرج features عبر actual_* ثم fallback على expected_*.
  - STRICT_MODE: إذا أي من critical values None → drop row (returns None)
  - encodes:
    - market_regime → int
    - session → int
  - X vector length = 12 (حسب ترتيب builder)
  - y = 1.0 if actual_pnl > 0 else 0.0

- `build_dataset_from_db(strict_mode=None) -> (X_train, y_train)`

---

### 4.22 `analysis/models/xgboost_v2_inference.py`
**المسؤولية:**
- inference عبر نموذج XGBoost v2 (feature order مطابق لمدخلات هذا الملف).

**الدوال:**
- `load_v2_model()`
- `get_session_now()`
- `predict_with_v2(...) -> {p_win, available}`
  - features (10 عناصر): rsi, atr, macd, trend_strength, trend_score, momentum_score, volatility_score, regime_enc, session_enc, direction_enc
- `should_trade_v2(p_win, threshold=0.60) -> bool`
- `get_size_multiplier(p_win) -> float`

---

### 4.23 `analysis/models/xgboost_trainer.py`
**المسؤولية:**
- تدريب نموذج XGBoost من `execution_dataset`.

**أهم الدالة:**
- `train_model_from_db(strict_mode=True, ...) -> Dict[str, Any]`

**مراحل التدريب داخل الكود:**
1. build_dataset_from_db(strict_mode)
2. تحديد task عبر `_infer_task_from_y(y)`
3. shuffle split (train/test)
4. `xgb.train` مع objective:
   - binary: logistic إذا classification
   - reg:squarederror إذا regression
5. save model versioned path: `models/xgb_model_v{timestamp}.json`
6. نسخ `version_path` إلى `models/xgb_model.json` (latest pointer)
7. evaluation عبر `evaluate_model`
8. كتابة feature importance + training_feature_stats

**دالة مساعدة موجودة:**
- `debug_dataset()` (تساعد على معرفة rows المرفوضة)

---

### 4.24 `train_pipeline.py`
**المسؤولية:**
- Pipeline تنفيذ التدريب بشكل strict.

**الدوال:**
- `build_dataset_strict(min_rows=50)`
  - يستخدم `validate_execution_row(r)` من `data_quality.py`
  - ثم `build_ml_row(r)`

- `main()`
  - يقوم بإعداد dataset strict ثم ينفذ `train_model_from_db(strict_mode=True, min_rows=...)`

---

### 4.25 `data_quality.py`
**المسؤولية:**
- التحقق strict لصف rows في execution_dataset.

**الدالة:**
- `validate_execution_row(row) -> (accepted, reasons)`
  - يضمن expected_* fields موجودة وغير None
  - يتحقق من ranges:
    - expected_rsi [0,100]
    - expected_atr > 0
    - expected_entry > 0
  - إذا status == closed: actual_pnl يجب أن يكون غير None

- `explain_rejected_row(row)`

---

### 4.26 `data/storage/database.py`
**المسؤولية:**
- طبقة SQLite: إنشاء الجداول + CRUD لتداولات/قرارات/dataset.

**جداول (Tables) الموجودة في init_db():**
1. `trades`
2. `execution_dataset`
3. `decisions`
4. `news`
5. `analysis`
6. `signals`
7. `daily_stats`
8. `weights_history`
9. `risk_events`
10. `performance`
11. `positions`

**دوال رئيسية:**
- `init_db()`
- `save_trade(...)`
- `close_trade_db(trade_id, pnl)`
- `close_trade_db_by_order_id(order_id, pnl)`
- `upsert_execution_expected(...)`
- `upsert_execution_actual(...)`
- `get_execution_dataset(order_id)`
- `get_open_trades()`
- `is_symbol_open(symbol, direction)`
- `get_total_open_trades()`
- `get_daily_stats()` / `update_daily_stats(pnl)`
- `save_decision(...)` / `get_last_decisions(limit)`
- `get_weights(symbol)` / `save_weights(symbol, weights)`
- dataset/positions sync:
  - `sync_position`, `get_synced_positions`, `clear_positions`
- performance:
  - `save_performance`, `get_recent_trades`, `get_symbol_performance`, `get_all_symbols_performance`, `get_best_worst_symbols`

> **تفصيل execution_dataset**: انظر القسم 10 لاحقاً.

---

### 4.27 `feedback/adaptive_weights.py`
**المسؤولية:**
- تعديل weights في Voting Engine بناءً على أداء المكونات.

**الدوال:**
- `get_component_accuracy(symbol, limit=20) -> float`
  - Accuracy = count(pnl>0)/count(rows) (للمكوّن) — **ملاحظة:** الكود لا يربط Accuracy بمكوّن بعينه بشكل صريح (فقط component list موجودة لكن نفس منطق count pnl используется لكل component). هذه حقيقة توثيقية.

- `update_weights(symbol)`
  - إذا recent trades < MIN_TRADES_TO_LEARN → return
  - current = get_weights(symbol)
  - target لكل component = accuracies[comp]/total_accuracy
  - smoothing يعتمد win_rate thresholds
  - normalizes weights sum
  - save_weights + save_performance

- `run_feedback_loop()`
  - لكل symbol: update_weights(symbol)
  - log_performance_summary()

---

### 4.28 `feedback/learning.py`
**المسؤولية:**
- learning cycle لتسجيل performance (لكن لا يربط weights بشكل مباشر).

**الدوال:**
- `learn_from_history(symbol)`
  - reads recent trades via get_recent_trades
  - uses calculate_metrics
  - save_performance

- `run_learning_cycle()`

---

### 4.29 `feedback/performance.py`
**المسؤولية:**
- حساب win_rate/profit_factor/sharpe من قائمة trades.

**الدالة:**
- `calculate_metrics(trades) -> dict`
- `analyze_performance(symbol) -> dict`

---

### 4.30 `telegram/telegram_bot.py`
**المسؤولية:**
- Telegram command handling (polling loop) + heartbeat.

**الدوال المهمة:**
- `start_telegram_bot(state)`
  - set_state
  - يطلق threads:
    - polling_loop
    - heartbeat_loop

**الأوامر COMMANDS:**
- /status /positions /balance /report /why /weights /dataset_status /stop /start /pause /emergency /news /performance /help /close

**ملاحظات توثيقية:**
- `cmd_analyze` يقرأ AI + trend/regime
- `cmd_dataset_status` يعتمد فقط على execution_dataset وليس على تقييم strict كامل.

---

### 4.31 `telegram/notifier.py`
**المسؤولية:**
- إرسال رسائل عبر Telegram.

**الدوال:**
- `send(text)`
- `notify_alert(msg)` / `notify_status(msg)` / `notify_start()`
- `notify_trade_opened(...)`
- `notify_trade_closed(...)`
- `notify_daily_report(stats)`

---

### 4.32 ملفات تم ذكرها لكن محتواها غير مقروء هنا (Not Found)
- `core/models.py` : **Not Found** (لكن imported classes: NewsItem, AINewsAnalysis, SentimentData, MultiTimeframeData)
- `core/exceptions.py` : **Not Found** (لكن RiskLimitError, QuantDingerAuthError, QuantDingerConnectionError imported)
- `utils/logger.py` : **Not Found**
- `data/market/client.py` و`data/market/hybrid_client.py` : **Not Found** (لكن main.py يستدعي get_indicators, get_price_hybrid, get_candles, get_indicators_hybrid)
- `decision/confidence_engine.py` : **Not Found** (مستدعى من main.py)
- `reports/report_generator.py` : **Not Found** (مذكور في main.py)
- `analysis/models/performance_monitor.py` : **Not Found** (مستدعى من reconciliation.py)
- `analysis/models/xgboost_inference.py` : **Not Found** (مستدعى في main.py ضمن استدعاء آخر غير مستخدم في مسار v2 كما يظهر)
- `data/news/calendar.py` : **Not Found** (main.py يستدعي is_high_impact_soon)
- `data.market.hybrid_client` : **Not Found**

---

## 5. Trading Workflow

### دورة التداول الكاملة (من بداية الدورة إلى إغلاق الصفقة)

#### A) بداية الدورة (main.py → run_cycle)
1. **MT5 check**
   - `check_mt5_status()`
   - إذا disconnected: reconnect via `connect_mt5()` أو skip.

2. **High impact news incoming**
   - `is_high_impact_soon(30)`
   - إذا True → notify_alert + return (cycle skipped).

3. **Fetch news**
   - `news = fetch_rss_news()`
   - إذا empty → return.
   - `news = filter_relevant_news(news)`

4. **Equity fetch**
   - equity = get_equity()

#### B) لكل symbol
1. **Filter symbol-specific news**
   - `filter_news_for_symbol(news, symbol)`

2. **AI news analysis (DeepSeek)**
   - `ai = analyze_news(symbol_news, symbol)`

3. **Signal generation**
   - `signal = generate_signal(ai)`

4. **Sentiment analysis**
   - `sentiment = analyze_sentiment(news, symbol)`
   - sentiment score: sent_score_val = sentiment.score إذا direction != neutral else 40

5. **MTF & technicals**
   - `mtf = get_multi_timeframe_analysis(symbol)`
   - `trend_score, trend_dir = get_trend_score(symbol)`
   - `momentum = get_momentum_score(symbol)`
   - `volatility_score = get_volatility_score(symbol)`
   - `regime = get_market_regime(symbol)`

6. **Voting decision**
   - `decision = make_decision(...)`
   - decision يحتوي:
     - direction (BUY/SELL/NEUTRAL)
     - final_score
     - ai_score, trend_score, momentum_score, sentiment_score, volatility_score
     - mtf_aligned، weights_used ...

7. **Confidence calculation**
   - `confidence = calculate_confidence(...)`
   - `calculate_confidence` content غير مقروء هنا (Not Found).

8. **Persist decision**
   - `save_decision(...)` → جدول decisions

9. **Skip criteria**
   - إذا direction == NEUTRAL أو not signal["is_valid"] → skip + save_decision(action=SKIP)

10. **Risk checks**
   - `ok, reason = can_trade(symbol, direction, final_score, ai_confidence, equity)`
   - إذا not ok → save_decision(action=SKIP) + continue

11. **ATR + SL/TP + Entry price**
   - `atr = get_atr(symbol)` (hybrid)
   - `entry_price = get_price_hybrid(symbol)`
   - إذا entry_price == 0 → skip
   - `sl, tp = calculate_sl_tp(symbol, entry_price, direction, atr)`

12. **Position sizing**
   - `size = calculate_position_size(equity, sl_distance, symbol, consecutive_losses=0, score=final_score)`

13. **Feature snapshot for ML**
   - indicators_data = get_indicators(symbol)
   - rsi/macd extracted
   - `features = build_trade_features(...)`

14. **ML Gate v2 (XGBoost)**
   - `v2_result = predict_with_v2(...)`
   - إذا not available → bypass (size_multiplier=1.0)
   - else إذا p_win < 0.60 → continue
   - else size_multiplier = get_size_multiplier(p_win)
   - size adjusted: `size = round(size * size_multiplier, 2)`

15. **Open trade**
   - `result = open_trade(symbol, direction, size, sl, tp, ai.reason[:80])`
   - إذا not success أو لا يوجد order_id → notify_alert

16. **Persist expected snapshot**
   - Builds expected_payload with expected_* fields
   - `required_non_null` list (لا يسمح None لبعض fields)
   - `missing = ...` إذا missing → skip open trade persistence (ملاحظة: هذا لا يراجع الصفقات التي تم فتحها؛ لكنه يمنع كتابة dataset)
   - `upsert_execution_expected(order_id, symbol, direction, expected_*, strategy="V3")`

17. **Persist trade**
   - `trade_id = save_trade(...)` مع order_id

18. **Notify Telegram**
   - notify_trade_opened(...)

#### C) إغلاق الصفقة (Reconciliation loop)
- `execution/reconciliation.py` يدير الإغلاق بناء على:
  1. Profit target: `PROFIT_TARGET_USD = 50`
  2. Trailing trigger/lock:
     - trigger at 50 * 0.50
     - locks 30% of profit
  3. News conflict:
     - analyze_news on fresh news
     - إذا bias يعاكس direction_norm وconfidence >= 0.75 → close

- عند الإغلاق:
  - `close_trade(order_id)` في QuantDinger
  - `close_trade_db_by_order_id(order_id, pnl)` في SQLite
  - `upsert_execution_actual(...)` لكتابة actual_* facts
  - notify_trade_closed

---

## 6. Decision Logic

### كيف يتم إنشاء الإشارة؟
**Signal Engine** (`decision/signal_engine.py`)
- Input: `AINewsAnalysis` من DeepSeek.
- Output: dict:
  - direction from bias
  - strength derived from impact_score/confidence
  - is_valid يساوي True فقط إذا:
    - confidence >= 0.60
    - bias في [bullish, bearish]
    - impact_score >= 55

### كيف يتم حساب الاتجاه (Direction)؟
**Voting Engine** (`decision/voting_engine.py`)
- يعتمد على votes من:
  - AI bias
  - Trend dir (from H4)
  - Momentum dir (from H1)
  - Sentiment score thresholds
- ثم يحدد direction:
  - BUY إذا bullish_votes > bearish_votes
  - SELL إذا bearish_votes > bullish_votes
  - NEUTRAL إذا التعادل

### كيف يعمل MTF؟
**Multi-Timeframe Analyzer** (`analysis/multi_timeframe/analyzer.py`)
- H4/H1/M15 → directions + alignment logic:
  - aligned strong: 3 non-neutral same
  - aligned moderate: 2+ same non-neutral أو H4==H1
  - aligned weak: H1==M15
  - else aligned False

### كيف يعمل Voting Score؟
- final_score computed من:
  - ai_score * ai_weight_boosted
  - trend_score * weights[trend]
  - momentum_score * weights[momentum]
  - sentiment_score * weights[sentiment]
  - volatility_score * weights[volatility]
- ثم multiplier based on dir_multiplier وMTF alignment.

### كيف يتم رفض/قبول الصفقة؟
Reject/Accept يتم على عدة بوابات:
1. Signal invalid أو direction NEUTRAL → skip
2. Risk Engine can_trade → reject
3. ML Gate v2:
   - if available and p_win < 0.60 → reject
4. Required non-null expected_* gating before dataset insert

---

## 7. Risk Management System

### Position sizing
- `risk/position_sizing.py`:
  - risk_percent depends on signal strength score and consecutive losses and equity thresholds.
  - size = (equity*risk_percent)/sl_distance
  - capped by MAX_LOT_PER_SYMBOL

### Daily loss limits
- `risk/risk_engine.py`:
  - daily_loss_pct = abs(stats.total_pnl)/starting_balance
  - إذا stats.total_pnl < 0 و daily_loss_pct >= 0.03 → reject

> ملاحظة توثيقية: `MAX_DAILY_LOSS` و `MAX_DAILY_LOSS_USD` موجودة في config.py لكن لم يظهر استخدامها داخل can_trade في الملف المقروء؛ can_trade يستخدم نسبة 3% ثابتة.

### Drawdown protection
- `risk/drawdown.py`:
  - halt_day عند daily_dd >= MAX_DRAWDOWN_HALT (5%)
  - half_risk عند >= ACCOUNT_DRAWDOWN_HALF (10%)
  - full_stop عند >= ACCOUNT_DRAWDOWN_STOP (20%)

### Correlation checks
- `risk/risk_engine.py`:
  - correlation groups بسيطة (USD_SHORT, USD_LONG)
  - إذا كانت هناك >=2 صفقات open في نفس group وفي نفس direction → reject

> ملاحظة توثيقية: config has MAX_CORRELATION and CORRELATION_LOOKBACK لكن لا يظهر استخدام مباشر في can_trade.

### Spread filters
- **Not Found**: لا يوجد منطق explicit “spread filter” ضمن الملفات التي تم فتحها. قد يكون في Not Found modules (مثل market clients) أو configuration أخرى.

### Volatility filters
- **Partial**: volatility يؤثر ضمن VotingEngine عبر `volatility_score` من indicators، و regime detection. لكن “فلتر” صريح يمنع التداول بسبب volatility غير ظاهر ضمن risk_engine المقروء.

### News filters
- News high impact pause:
  - `is_high_impact_soon(30)` — **Not Found** content.
- News conflict exit inside reconciliation:
  - news analyzer vs current position direction.

---

## 8. News & AI System

### مصادر الأخبار
- RSS feeds list في `data/news/fetcher.py` (Reuters, Forexlive, DailyFX, FXStreet, Kitco, BBC, Investing, Marketwatch, CNBC, Yahoo, Goldprice, …)
- Finnhub news endpoint.

### DeepSeek
- prompt rules موجودة في `analysis/ai/deepseek.py`:
  - return ONLY valid JSON
  - bias neutral if unclear
  - impact_score <40 weak

### Sentiment Analysis
- `analysis/sentiment/analyzer.py`:
  - keyword matching على headlines/summary
  - score calculation عبر bullish_ratio

### Impact Score
- DeepSeek output: impact_score.
- In deepseek.py: news_impact_score = impact_score.

### Confidence
- DeepSeek output confidence (0..1)
- Risk engine وML gate يستخدمان confidence/ai_confidence.

### Bias
- DeepSeek bias = bullish/bearish/neutral.
- Signal engine يترجم bias → direction.

### تدفق البيانات بينها (Data Flow)
- fetcher.py → list[NewsItem]
- scoring/filter_relevant_news → ordered top list
- analyze_news(news, symbol) → AINewsAnalysis
- analyze_sentiment(news, symbol) → SentimentData
- make_decision uses:
  - ai_analysis.impact_score, ai_analysis.bias, ai_analysis.confidence
  - sentiment.score
  - technical & mtf inputs

---

## 9. Multi Timeframe Analysis

### H4
- TF_TREND = "H4" في config.
- `analysis/technical/indicators.get_trend_score` يحلل ma_trend/RSI.

### H1
- TF_DECISION = "H1" في config.
- `get_momentum_score(symbol)` default timeframe=TF_DECISION.

### M15
- TF_TIMING = "M15" في config.
- `get_multi_timeframe_analysis` يستدعي momentum_score على timeframe=TF_TIMING.

### كيف يتم استخدامها فعلياً داخل المشروع؟
- alignment logic في `analysis/multi_timeframe/analyzer.py`:
  - produces mtf.aligned and mtf.strength
- voting_engine يستخدم mtf_data.aligned/strength لتعديل final_score.

### مشاكل/تناقضات تم اكتشافها (بدون إصلاح)
1. **Not Found potential mismatch**: هناك تعارض محتمل بين مصطلحات “direction” في بعض الأماكن (BUY/SELL) وبين normalize type in reconciliation (buy/sell)؛ تم توثيق normalization في reconciliation.
2. **MTF strength labels**: Voting engine يعامل `mtf_strength` كقيمة string ويستخدم "strong" فقط لرفع/خفض كبير، بينما weak/moderate له قيم تؤثر بشكل مختلف.

---

## 10. Database Documentation

### نظرة عامة
DB هي SQLite في `config.DB_FILE = "trading_bot_v3.db"`.
- WAL mode مفعّل
- foreign_keys=ON

### execution_dataset (طلبك التفصيلي)

#### الغرض العام
`execution_dataset` هي **حجر الأساس للـ ML**.
- قبل فتح الصفقة: يتم كتابة **expected snapshot** للحظة القرار
- عند الإغلاق: reconciliation يكتب **actual facts** (entry/exit/pnl/slippage/quality)

#### اسم الجدول
- `execution_dataset`

#### الأعمدة (Columns) ووظيفة كل عمود
> الأنواع هنا كما تظهر في CREATE TABLE (SQLite لا يفرض strict typing).

**dataset_created_at**
- النوع: TEXT
- الغرض: وقت إنشاء dataset row
- من يكتبه: `upsert_execution_expected` عند INSERT
- من يقرأه: `ml_dataset_builder` وtrainers (بشكل غير مباشر)

**dataset_updated_at**
- النوع: TEXT
- الغرض: آخر تحديث في expected أو actual
- من يكتبه: UPSERT on conflict

**order_id**
- النوع: TEXT UNIQUE
- الغرض: مفتاح ربط بين QuantDinger/MT5 order وبين expected/actual
- من يكتبه: `main.py` عبر `upsert_execution_expected` + reconciliation عبر `upsert_execution_actual`
- من يقرأه: `get_execution_dataset` في reconciliation + ml builders.

**symbol** / **direction**
- النوع: TEXT
- الغرض: تعريف السياق

---

### Expected snapshot (open-time features)

**expected_entry**
- النوع: REAL
- الغرض: entry price المتوقع (في هذا الكود: يُمرر قيمة entry_price التي تم الحصول عليها بعد فتح الصفقة)
- من يكتبه: `main.py` via upsert_execution_expected
- من يقرأه: `data_quality.validate_execution_row`, `ml_dataset_builder.build_ml_row`, feature gating in training.

**expected_final_score**
- النوع: REAL
- الغرض: final_score من VotingEngine (normalized/managed في main)

**expected_rsi**
- النوع: REAL
- الغرض: RSI ضمن snapshot

**expected_macd**
- النوع: REAL
- الغرض: MACD

**expected_session**
- النوع: TEXT
- الغرض: session label (قد يكون نصي)

**expected_spread**
- النوع: REAL
- الغرض: spread snapshot (main يضع spread=0.0 في snapshot)

**expected_atr**
- النوع: REAL
- الغرض: ATR snapshot

**expected_trend_strength**
- النوع: REAL
- الغرض: strength (normalization/encoding)

**expected_momentum_score**
- النوع: REAL
- الغرض: momentum_score

**expected_volatility_score**
- النوع: REAL
- الغرض: volatility_score

**expected_market_regime**
- النوع: TEXT
- الغرض: regime label

**expected_ai_score**
- النوع: REAL
- الغرض: ai_score

**expected_sentiment_score**
- النوع: REAL

**expected_news_impact_score**
- النوع: REAL

**expected_ai_confidence**
- النوع: REAL
- الغرض: ai_confidence

**expected_trend_score**
- النوع: REAL

**expected_momentum_score_legacy / expected_sentiment_score_legacy / expected_volatility_score_legacy**
- النوع: REAL
- الغرض: backward compatibility columns

**expected_indicators_json**
- النوع: TEXT
- الغرض: JSON لمؤشرات إضافية (لكن في main يتم تمرير None)

**status**
- النوع: TEXT DEFAULT 'open'
- الغرض: حالة صف
- من يكتبها: upsert expected/actual (expected keeps 'open', actual writes 'closed')

---

### Actual facts (close-time labels)

**actual_entry**
- النوع: REAL
- الغرض: actual entry (price_open من QD)
- من يكتبه: reconciliation.upsert_execution_actual

**actual_exit**
- النوع: REAL
- الغرض: actual exit (price_current من QD)

**actual_pnl**
- النوع: REAL
- الغرض: label التدريب (win/loss)

**actual_rsi / actual_macd / actual_session / actual_spread / actual_atr**
- النوع: REAL/TEXT
- الغرض: actual indicators at close
- من يكتبها: reconciliation حالياً تمرير None لactual_indicators_json وفي الكود الحالي أيضاً يمرر actual_* غالباً None لأن upsert_execution_actual signature لا يضع هذه الأعمدة في INSERT (فقط actual_entry/exit/pnl وحقول slippage وجودة...)
  - **تبعاً للكود المقروء:** reconciliation.upsert_execution_actual لا يمرر actual_rsi/actual_macd/… → هذه الأعمدة غير populated تلقائياً (ستبقى NULL).

**actual_trend_strength / actual_momentum_score / actual_volatility_score / actual_market_regime**
- النوع: REAL/TEXT
- الغرض: actual-derived features عند close

**actual_ai_score / actual_sentiment_score / actual_news_impact_score**
- النوع: REAL

**spread_at_entry**
- النوع: REAL

**slippage**
- النوع: REAL

**execution_delay_ms**
- النوع: INTEGER

**execution_quality_score**
- النوع: REAL
- الغرض: جودة تنفيذ (computed في reconciliation). في reconciliation يتم ضمانه غير NULL قدر الإمكان:
  - إذا slippage is None → 0.0 (ليس None)

**price_gap**
- النوع: REAL
- الغرض: actual_entry - expected_entry (إن أمكن)

**actual_indicators_json**
- النوع: TEXT

**status**
- open|closed|orphaned (تعريف مبني في schema)

> نقطة توثيقية مهمة: في `ml_dataset_builder.build_ml_row` يتم بناء X عبر actual_* أولاً ثم fallback على expected_*. لذلك رغم NULLs في actual_* غالباً، يبقى training ممكن باستخدام expected_*.

---

## 11. Machine Learning System

### وصف عام
ML Gate (v2) يستخدم `analysis/models/xgboost_v2_inference.py` في main.
أما التدريب فهو:
- dataset builder: `analysis/features/ml_dataset_builder.py`
- quality validation: `data_quality.py`
- training: `analysis/models/xgboost_trainer.py`
- pipeline: `train_pipeline.py`

### الملفات المطلوبة في التوثيق (كما طلبت)

#### `train_pipeline.py`
- كما تم قراءته: يعتمد validate_execution_row strict ثم build_ml_row ثم train_model_from_db.

#### `data_quality.py`
- validate_execution_row يرفض rows التي فيها expected_* غير موجودة/None أو خارج ranges.

#### `analysis/features/ml_dataset_builder.py`
- STRICT_MODE = True
- build_ml_row يسقط الصفوف إذا أي critical feature في critical_values None.
- X vector length = 12.
- y = 1 إذا actual_pnl > 0.

#### `analysis/models/xgboost_trainer.py`
- trains XGBoost ويكتب:
  - versioned model: xgb_model_v{timestamp}.json
  - latest pointer: models/xgb_model.json
- يكتب feature_importance.json
- يكتب training_feature_stats.json (training stats)

#### `analysis/models/xgboost_v2_inference.py`
- inference model v2
- features length=10
- output p_win + available

---

### كيف يتم بناء dataset؟
- source: `execution_dataset`.
- `ml_dataset_builder.build_dataset_from_db`:
  - يقرأ كل rows من table.
  - build_ml_row لكل row:
    - extracts features from actual_* fallback expected_*
    - STRICT_MODE: إذا أي critical None → drop
    - encodes market_regime/session
    - returns X and y

### كيف يتم التدريب؟
- `train_pipeline.main`:
  - يقوم بتجميع strict accepted dataset.
  - ثم `train_model_from_db(strict_mode=True, min_rows=50)`
- `xgboost_trainer.train_model_from_db`:
  - يحسم task: classification إذا y binary {0,1}
  - else regression
  - split train/test
  - xgb.train objective binary:logistic أو reg:squarederror

### كيف يتم التنبؤ؟
- في التشغيل: `main.py` يستخدم `predict_with_v2`.
- يتم قبول الدخول بناء على should_trade_v2(p_win, threshold=0.60).

---

## 12. Closed-Loop Learning Architecture

### كيف تنتقل البيانات من الصفقة إلى التدريب؟
**1) فتح صفقة (Expected write)**
- `main.py` بعد `open_trade` ينفذ:
  - `upsert_execution_expected(order_id, expected_*)`
- هذا يخلق snapshot للحظة القرار.

**2) إغلاق صفقة (Actual write)**
- `execution/reconciliation.py` بعد close:
  - يستدعي `upsert_execution_actual(order_id, actual_entry, actual_exit, actual_pnl, slippage, execution_quality_score, ...)`
  - ثم `close_trade_db_by_order_id(order_id, pnl)`

**3) Training dataset build**
- `analysis/features/ml_dataset_builder.py`:
  - X uses actual_* if present else expected_*
  - y uses actual_pnl

**4) training loop**
- `analysis/models/system_orchestrator.py` أو `train_pipeline.py`:
  - يقرر should_retrain
  - يطبق strict mode وتدريب XGBoost
  - تحديث latest pointer: models/xgb_model.json

**5) Decision loop جديد**
- `main.py` يستخدم inference v2 ويضبط size.

---

## 13. Configuration Reference (config.py)

> توثيق المتغيرات المهمة كما ظهرت.

### API Keys
- `DEEPSEEK_API_KEY`
- `FINNHUB_API_KEY`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

### QuantDinger
- `QUANTDINGER_URL`
- `QUANTDINGER_USERNAME`
- `QUANTDINGER_PASSWORD`

### MT5
- `MT5_LOGIN` (int)
- `MT5_PASSWORD`
- `MT5_SERVER`
- `MT5_PATH`

### Trading pairs
- `SYMBOLS = ["EURUSD", "XAUUSD", "GBPUSD"]`

### Timeframes
- `TF_TREND = "H4"`
- `TF_DECISION = "H1"`
- `TF_TIMING = "M15"`

### Risk Management
- `BASE_RISK_PERCENT`
- `MAX_DAILY_LOSS`
- `MAX_DAILY_LOSS_USD`
- `MAX_DRAWDOWN_HALT`
- `ACCOUNT_DRAWDOWN_HALF`
- `ACCOUNT_DRAWDOWN_STOP`
- `MAX_OPEN_TRADES`
- `STOP_AFTER_LOSSES`
- `MAX_SL_PIPS`
- `ATR_SL_MULTIPLIER`
- `ATR_TP_MULTIPLIER`

### Decision Voting
- `INITIAL_WEIGHTS = {ai, trend, momentum, sentiment, volatility}`
- `MIN_SCORE`
- `AI_MIN_CONFIDENCE`

### News
- `NEWS_SOURCE_WEIGHTS`
- `NEWS_DECAY_HOURS`
- `NEWS_CHECK_INTERVAL`

### Execution
- `MAX_RETRIES`
- `RETRY_DELAY`
- `ORDER_TIMEOUT` (Not Found usage in opened modules)

### Feedback / Learning
- `WEIGHT_LEARNING_RATE` (Not Found usage in adaptive_weights code)
- `MIN_TRADES_TO_LEARN`
- `WEIGHT_SMOOTHING`
- `FEEDBACK_BATCH_SIZE`

### Reconciliation
- `RECONCILIATION_INTERVAL` (Not Found usage; start_reconciliation uses hardcoded interval=60 in reconciliation thread)

### Watchdog
- `WATCHDOG_INTERVAL`
- `WATCHDOG_FAIL_LIMIT`

### Correlation
- `MAX_CORRELATION` (Not Found usage in risk_engine code)
- `CORRELATION_LOOKBACK` (Not Found usage in risk_engine code)

### DB
- `DB_FILE = "trading_bot_v3.db"`

### PIP values
- `PIP_VALUES` mapping per symbol.

---

## 14. Logging & Monitoring

### نظام اللوق
- جميع الملفات المقروءة تستخدم `utils.logger.get_logger`.
- content `utils/logger.py` **Not Found**.

### التتبع (Monitoring)
- Telegram heartbeat loop يرسل نبض كل 900 ثانية.
- Reconciliation logs في reconciliation.py.

### Telegram
- `telegram/notifier.py` يرسل:
  - notify_start
  - notify_trade_opened
  - notify_trade_closed
  - notify_daily_report
- `telegram/telegram_bot.py`:
  - polling loop لمعالجة الأوامر
  - heartbeat loop لإرسال state دوري

---

## 15. Current Issues Found

> قسم “مكتشفات” بدون إصلاح.

1. **تكرار تعريف login في `execution/quantdinger_client.py`**
   - توجد دالتان باسم `login()` داخل نفس الملف (كما ظهر في النص المقروء).
   - Python سيستخدم آخر تعريف فقط، ما قد يسبب اختلافاً غير مقصود.

2. **ML feature/normalization inconsistency risk**
   - `feature_builder.build_trade_features` يقوم ب normalization لـ عدة features (0-1 / [-1,1]).
   - بينما `xgboost_v2_inference.predict_with_v2` يتوقع features مختلفة (10 features) ويستخدم encodings/thresholds مع `trend_score` و `trend_strength` و `momentum_score`.
   - في `main.py` يتم تمرير `macd` إلى predict_with_v2 مباشرة، و`momentum_score` يتم استخراج جزء من tuple أو value.
   - هذا قد يؤدي لتفاوت scale بين training v2 و inference v2.

3. **Risk engine correlation groups مبسطة**
   - لا يتم استخدام MAX_CORRELATION ولا CORRELATION_LOOKBACK داخل الكود المقروء.
   - correlation logic يعتمد على نفس direction_count داخل group فقط.

4. **Daily loss calculation approximate**
   - can_trade يستخدم starting_balance = equity - total_pnl لتقدير daily loss pct.
   - قد ينتج سلوكاً غير متوقع إذا daily stats total_pnl لا تمثل “yesterday vs today” بشكل دقيق.

5. **Sentiment keyword filtering قد يضعف coverage**
   - sentiment analyzer يفلتر relevant_news بحسب SYMBOL_KEYWORDS في حال توفر كلمات.
   - إذا لا يوجد match قد يستخدم headlines من news_list[:5] (fallback).

6. **Reconciliation: upsert_execution_actual signature لا يتضمن actual_* features كاملة**
   - upsert_execution_actual يكتب فقط أعمدة مثل actual_entry/exit/pnl/slippage/execution_quality_score/price_gap/actual_indicators_json.
   - أعمدة actual_rsi/actual_macd/... غير مملوءة في الاستدعاءات الحالية.
   - training يعتمد fallback على expected_* في ml_dataset_builder.

7. **adaptive_weights accuracy mapping لا يربط component بميزة فعلية**
   - get_component_accuracy يستخدم pnl/direction من trades فقط دون فصل component contributions.

8. **ملفات multi-layer مهمّة غير مقروءة نصياً ضمن هذه الجولة**
   - `core.models`, `core.exceptions`, `utils.logger`, `analysis/decision/confidence_engine`, `data/news/calendar`, `data/market/*`, `reports/*`, `analysis/models/performance_monitor`.
   - وجود “وظيفة” كاملة لا يمكن الجزم بها هنا.

---

## 16. Technical Debt

1. **تكرار دالة login** في quantdinger_client.

2. **عدم وضوح contract بين expected_* وactual_* وبين feature_builder/ml_dataset_builder**
   - بعض columns في execution_dataset غير مملوءة في reconciliation، بينما builder/validation يفترض critical expected_*.

3. **انعدام توحيد scale بين xgboost_v2_inference و xgboost_trainer**
   - v2 inference uses 10-feature vector مع encodings.
   - trainer uses 12-feature vector via ml_dataset_builder.
   - هذا مقبول إذا التدريب v2 منفصل وبنفس schema، لكن غير موثّق هنا.

4. **تداخل parsing logic مع inference logic في main.py**
   - main.py يبني expected_payload يضمن non-null gating لعدد من expected fields.

5. **اعتماد heavy network calls في نفس دورة main.py**
   - deepseek لكل symbol لكل cycle.

---

## 17. Final System Assessment

### نقاط القوة (Strengths)
- فصل طبقات واضح: News → AI/Sentiment → MTF/Technicals → Voting → Risk → ML Gate → Execution.
- وجود DB-first architecture عبر `execution_dataset` لتغذية التدريب.
- Reconciliation يعالج الحالات الواقعية (profit target + trailing + news conflict).
- ML Gate v2 موجود كطبقة إضافية لتصفية صفقات منخفضة `p_win`.
- Telegram يوفر مراقبة وتشغيل أوامر.

### نقاط الضعف (Weaknesses)
- بعض المسارات تعتمد modules غير مقروءة هنا (confidence_engine، calendar، market clients…)، ما يجعل توثيق بعض “العقود” غير كامل.
- يوجد تكرار واضح في QuantDinger client login.
- adaptive_weights لا يميّز component accuracies بشكل حقيقي بحسب code الحالي.

### الجاهزية للتشغيل الحقيقي (Production Readiness)
- **متوسطة**.
- الأسباب:
  - هناك watchdog ومراقبة Reconciliation.
  - لكن ما يزال هناك “unknown contracts” في modules غير مقروءة، وتكرار login في QD client.
  - بيانات expected/actual dataset تستخدم fallback على expected_*؛ هذا جيد للتدريب لكن قد يخفف دلالة actual_*.

### الجاهزية للتدريب (Training Readiness)
- **جيدة** من ناحية بنية DB + strict validation + dataset builder.
- لكن تعتمد جودة expected_* المكتوبة من main.py على non-null gating.

### الجاهزية للتوسع (Scalability)
- **متوسطة إلى ضعيفة**:
  - cycle loop ينفذ تحليل DeepSeek لكل symbol وقد يكون مكلفاً.
  - training orchestrator يعتمد approximate new_rows_count.
  - DB queries بدون index مخصص غير موثّق هنا (Not Found: هل تم عمل indices؟) — init_db يعرّف UNIQUE على order_id فقط.

---

## ملاحق (Appendix) — خريطة مرجعية سريعة للملفات

- التشغيل:
  - `main.py`
  - `execution/quantdinger_client.py`
  - `execution/reconciliation.py`
  - `execution/mt5_watchdog.py`
  - `telegram/telegram_bot.py` + `telegram/notifier.py`

- القرار:
  - `analysis/ai/deepseek.py`
  - `analysis/sentiment/analyzer.py`
  - `analysis/technical/indicators.py`
  - `analysis/technical/regime.py`
  - `analysis/multi_timeframe/analyzer.py`
  - `decision/voting_engine.py`
  - `decision/signal_engine.py`
  - `decision/confidence_engine.py` (**Not Found**)

- المخاطر:
  - `risk/risk_engine.py`
  - `risk/position_sizing.py`
  - `risk/sltp.py`
  - `risk/drawdown.py`

- ML:
  - `analysis/features/feature_builder.py`
  - `analysis/features/ml_dataset_builder.py`
  - `analysis/models/xgboost_trainer.py`
  - `analysis/models/xgboost_v2_inference.py`
  - `analysis/models/system_orchestrator.py`
  - `train_pipeline.py`

- DB:
  - `data/storage/database.py`

---

