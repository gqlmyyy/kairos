# Trading Bot V3 - decision/voting_engine.py
# Weighted voting system: AI + Trend + Momentum + Sentiment + Volatility

from utils.logger import get_logger
from data.storage.database import get_weights
from config import INITIAL_WEIGHTS

logger = get_logger("voting_engine")

def make_decision(
   symbol: str,
   ai_analysis: dict,
   trend_data: dict,
   momentum_data: tuple,
   volatility_score: float,
   sentiment_score_val: float,
   mtf_data: dict
) -> dict:
   weights = get_weights(symbol)

   ai_score = ai_analysis.get("impact_score", 0)
   ai_bias = ai_analysis.get("bias", "neutral")
   ai_confidence = ai_analysis.get("confidence", 0)

   trend_score = trend_data.get("h4_score", 40)
   trend_dir = trend_data.get("h4_direction", "neutral")

   momentum_score_val, momentum_dir = momentum_data

   # ==============================
   # 1. التصويت على الاتجاه
   # ==============================
   votes = {
       "bullish": 0,
       "bearish": 0
   }

   # AI يأخذ وزن مضاعف بناءً على الثقة
   ai_vote_weight = 1.5 if ai_confidence >= 0.85 else 1.0
   if ai_bias == "bullish":
       votes["bullish"] += ai_vote_weight
   elif ai_bias == "bearish":
       votes["bearish"] += ai_vote_weight

   # Trend
   if trend_dir == "bullish":
       votes["bullish"] += 1.0
   elif trend_dir == "bearish":
       votes["bearish"] += 1.0

   # Momentum
   if momentum_dir == "bullish":
       votes["bullish"] += 0.75
   elif momentum_dir == "bearish":
       votes["bearish"] += 0.75

   # Sentiment
   if sentiment_score_val >= 70:
       votes["bullish"] += 0.5
   elif sentiment_score_val <= 30:
       votes["bearish"] += 0.5

   total_votes = votes["bullish"] + votes["bearish"]

   if votes["bullish"] > votes["bearish"]:
       final_direction = "BUY"
       dir_multiplier = votes["bullish"] / total_votes if total_votes > 0 else 0
   elif votes["bearish"] > votes["bullish"]:
       final_direction = "SELL"
       dir_multiplier = votes["bearish"] / total_votes if total_votes > 0 else 0
   else:
       final_direction = "NEUTRAL"
       dir_multiplier = 0

   # إجماع كامل → مكافأة
   if dir_multiplier >= 0.85:
       dir_multiplier *= 1.2

   # ==============================
   # 2. حساب الـ Score
   # ==============================
   # AI confidence boost
   ai_weight_boosted = weights["ai"] * (1 + ai_confidence * 0.3)

   final_score = (
       ai_score            * ai_weight_boosted +
       trend_score         * weights["trend"] +
       momentum_score_val  * weights["momentum"] +
       sentiment_score_val * weights["sentiment"] +
       volatility_score    * weights["volatility"]
   ) * dir_multiplier

   # ==============================
   # 3. تعديل MTF
   # ==============================
   mtf_aligned = mtf_data.get("aligned", False)
   mtf_strength = mtf_data.get("strength", "weak")

   if not mtf_aligned:
       if mtf_strength == "strong":
           final_score *= 0.75   # معاكس قوي → خصم أقل
           logger.info(f"{symbol}: MTF strongly misaligned, reducing score 25%")
       else:
           final_score *= 0.90   # معاكس ضعيف → خصم أقل
           logger.info(f"{symbol}: MTF weakly misaligned, reducing score 10%")
   else:
       if mtf_strength == "strong":
           final_score *= 1.10   # متوافق قوي → مكافأة
           logger.info(f"{symbol}: MTF strongly aligned, boosting score 10%")

   final_score = round(min(final_score, 100), 1)

   return {
       "direction": final_direction,
       "final_score": final_score,
       "ai_score": ai_score,
       "ai_confidence": ai_confidence,
       "trend_score": trend_score,
       "momentum_score": momentum_score_val,
       "sentiment_score": sentiment_score_val,
       "volatility_score": volatility_score,
       "weights_used": weights,
       "mtf_aligned": mtf_aligned,
       "dir_multiplier": dir_multiplier,
       "reason": ai_analysis.get("reason", "")
   }