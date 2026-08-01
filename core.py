"""
core.py

موتور اصلی پروژه AI Excel Analyst.
شامل تمام توابع لازم برای:
1. خلاصه‌سازی دیتاست خام (generate_llm_data_summary)
2. ساخت خودکار KPI (build_auto_kpis)
3. تبدیل KPI به متن قابل فهم برای LLM (generate_kpi_summary_text)
4. گرفتن Insight نهایی از مدل (get_llm_insights)

این فایل هیچ‌وقت مستقیم اجرا نمی‌شود؛ فقط توسط cli.py یا نوت‌بوک import می‌شود.
"""

import os
import time
import pandas as pd
import numpy as np
from google import genai
from google.genai import types
from google.genai import errors as genai_errors



from dotenv import load_dotenv
load_dotenv() # بارگذاری متغیرها در core.py
# =========================================================
# 1. خلاصه دیتاست خام
# =========================================================
def generate_llm_data_summary(
    df: pd.DataFrame,
    dataset_name: str = "Dataset",
    sample_n: int = 5,
    max_categories: int = 5,
    high_cardinality_threshold: float = 0.9,
    max_chars: int = 6000,
) -> str:
    """
    یک خلاصه‌ی متنی کامل و خوانا از یک DataFrame تولید می‌کند که برای دادن
    به یک LLM مناسب است (بدون نیاز به دیدن داده خام).
    """
    lines = []
    lines.append(f"# خلاصه دیتاست: {dataset_name}")
    lines.append("")

    n_rows, n_cols = df.shape
    n_duplicates = df.duplicated().sum()
    memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

    lines.append("## اطلاعات کلی")
    lines.append(f"- تعداد سطرها: {n_rows:,}")
    lines.append(f"- تعداد ستون‌ها: {n_cols}")
    lines.append(f"- سطرهای تکراری: {n_duplicates:,} ({100 * n_duplicates / max(n_rows, 1):.2f}%)")
    lines.append(f"- حجم حافظه (تقریبی): {memory_mb:.2f} مگابایت")
    lines.append("")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c not in datetime_cols]

    lines.append("## ستون‌ها و مقادیر گمشده")
    for col in df.columns:
        dtype = df[col].dtype
        n_missing = df[col].isna().sum()
        pct_missing = 100 * n_missing / max(n_rows, 1)
        n_unique = df[col].nunique(dropna=True)
        flag = "  ⚠️ ستون ثابت" if n_unique <= 1 else ""
        lines.append(f"- **{col}** ({dtype}): {n_missing:,} گمشده ({pct_missing:.1f}%), {n_unique:,} یکتا{flag}")
    lines.append("")

    if numeric_cols:
        lines.append("## آمار ستون‌های عددی")
        desc = df[numeric_cols].describe().T
        # محاسبه skew با مدیریت خطا برای دیتاست‌های بزرگ
        with np.errstate(all='ignore'):
            desc["skew"] = df[numeric_cols].skew()
        for col, row in desc.iterrows():
            lines.append(
                f"- **{col}**: min={row['min']:.3g}, max={row['max']:.3g}, "
                f"mean={row['mean']:.3g}, median={row['50%']:.3g}, std={row['std']:.3g}, skew={row['skew']:.2f}"
            )
        lines.append("")

        if len(numeric_cols) > 1:
            # محاسبه همبستگی فقط برای ستون‌های عددی با تنوع کافی
            valid_num_cols = [col for col in numeric_cols if df[col].nunique() > 1]
            if len(valid_num_cols) > 1:
                corr = df[valid_num_cols].corr()
                strong_pairs = []
                for i, c1 in enumerate(valid_num_cols):
                    for c2 in valid_num_cols[i + 1:]:
                        r = corr.loc[c1, c2]
                        if abs(r) >= 0.6 and not np.isnan(r):
                            strong_pairs.append((c1, c2, r))
                if strong_pairs:
                    lines.append("### همبستگی‌های قابل توجه (|r| >= 0.6)")
                    for c1, c2, r in sorted(strong_pairs, key=lambda x: -abs(x[2])):
                        lines.append(f"- {c1} ↔ {c2}: r = {r:.2f}")
                    lines.append("")

    if datetime_cols:
        lines.append("## ستون‌های تاریخ/زمان")
        for col in datetime_cols:
            min_val = df[col].min()
            max_val = df[col].max()
            if pd.notna(min_val) and pd.notna(max_val):
                lines.append(f"- **{col}**: از {min_val} تا {max_val}")
        lines.append("")

    if categorical_cols:
        lines.append("## ستون‌های دسته‌ای")
        for col in categorical_cols:
            n_unique = df[col].nunique(dropna=True)
            ratio = n_unique / max(n_rows, 1)
            if ratio >= high_cardinality_threshold and n_unique > max_categories:
                lines.append(f"- **{col}**: {n_unique:,} یکتا از {n_rows:,} سطر → احتمالاً شناسه (ID)")
            else:
                top_vals = df[col].value_counts(dropna=True).head(max_categories)
                top_str = ", ".join(f"{k}: {v:,}" for k, v in top_vals.items())
                lines.append(f"- **{col}** ({n_unique:,} یکتا): پرتکرارترین‌ها → {top_str}")
        lines.append("")

    # نمونه از داده
    sample_df = df.sample(min(sample_n, n_rows), random_state=42) if n_rows > 0 else df
    lines.append(f"## نمونه‌ای از {len(sample_df)} سطر داده")
    # نمایش بهتر برای ستون‌های طولانی
    with pd.option_context('display.max_colwidth', 30, 'display.width', 100):
        lines.append(sample_df.to_string(max_cols=20))
    lines.append("")

    result = "\n".join(lines)
    if len(result) > max_chars:
        # اگر طول زیاد بود، نمونه را حذف کن
        sample_index = None
        for i, line in enumerate(lines):
            if line.startswith("## نمونه‌ای از"):
                sample_index = i
                break
        if sample_index is not None:
            lines = lines[:sample_index]
            lines.append("## نمونه سطرها")
            lines.append("(به دلیل محدودیت طول، نمونه سطرها حذف شد)")
            result = "\n".join(lines)

    return result


# =========================================================
# 2. ساخت خودکار KPI
# =========================================================
def _find_column(df: pd.DataFrame, keywords: list, include_persian: bool = True) -> str | None:
    """
    جستجوی نام ستون بر اساس کلیدواژه (case-insensitive).
    اگر include_persian=True، کلیدواژه‌های فارسی معادل را هم جستجو می‌کند.
    """
    persian_keywords_map = {
        'revenue': ['درآمد', 'فروش', 'مبلغ', 'قیمت', 'فروش خالص'],
        'sales': ['فروش', 'فروش', 'مبلغ', 'درآمد'],
        'amount': ['مبلغ', 'مقدار', 'ارزش'],
        'total': ['جمع', 'کل', 'مجموع'],
        'price': ['قیمت', 'بهای', 'ارزش'],
        'order': ['سفارش', 'شماره سفارش', 'کد سفارش'],
        'customer': ['مشتری', 'خریدار', 'کاربر'],
        'date': ['تاریخ', 'زمان', 'تاریخ ثبت'],
        'created': ['ایجاد', 'ثبت', 'ساخت'],
        'time': ['زمان', 'ساعت', 'تاریخ']
    }
    
    # ساخت کلیدواژه‌های نهایی
    all_keywords = keywords.copy()
    if include_persian:
        for kw in keywords:
            if kw in persian_keywords_map:
                all_keywords.extend(persian_keywords_map[kw])
            # اضافه کردن معادل‌های فارسی
            persian_equiv = {
                'revenue': ['درآمد', 'فروش', 'مبلغ', 'قیمت'],
                'sales': ['فروش', 'فروش‌ها', 'مبلغ فروش'],
                'amount': ['مبلغ', 'مقدار', 'ارزش'],
                'total': ['جمع', 'کل', 'مجموع', 'کل مبلغ'],
                'price': ['قیمت', 'بهای', 'نرخ'],
                'order_id': ['شماره سفارش', 'کد سفارش', 'شناسه سفارش'],
                'customer': ['مشتری', 'خریدار', 'کاربر', 'شناسه مشتری'],
                'date': ['تاریخ', 'تاریخ ثبت', 'تاریخ ایجاد'],
                'created': ['ایجاد', 'ثبت', 'ساخت'],
                'time': ['زمان', 'ساعت', 'تاریخ']
            }
            if kw in persian_equiv:
                all_keywords.extend(persian_equiv[kw])
    
    # جستجو در نام ستون‌ها
    for col in df.columns:
        col_lower = str(col).lower().strip()
        # ستون ممکن است فارسی یا انگلیسی باشد
        if any(kw in col_lower for kw in all_keywords):
            return col
        # همچنین چک کردن اگر ستون انگلیسی و کلیدواژه فارسی باشد
        for kw in all_keywords:
            if kw in col_lower:
                return col
    return None


def build_auto_kpis(df: pd.DataFrame, date_col: str | None = None):
    """
    به‌صورت خودکار تلاش می‌کند KPIهای رایج کسب‌وکاری را از ستون‌های
    موجود در df بسازد. اگر ستون خاصی پیدا نشود، آن KPI ساخته نمی‌شود.

    خروجی: (kpi_df, kpi_definitions)
    """
    kpi_definitions = {}
    
    # جستجوی ستون‌های کلیدی با پشتیبانی از فارسی
    revenue_col = _find_column(df, ["revenue", "sales", "amount", "total","Price", "price",'درآمد'])
    order_col = _find_column(df, ["order_id", "order", "invoice", "transaction", "شماره سفارش"])
    customer_col = _find_column(df, ["customer", "client", "user_id", "user", "مشتری"])
    
    if date_col is None:
        date_col = _find_column(df, ["date", "created", "time", "تاریخ"])

    work_df = df.copy()

    # تنظیم ستون دوره
    if date_col and date_col in work_df.columns:
        try:
            work_df[date_col] = pd.to_datetime(work_df[date_col], errors="coerce")
            # حذف ردیف‌هایی که تاریخ نامعتبر دارند
            work_df = work_df.dropna(subset=[date_col])
            if not work_df.empty:
                work_df["_period"] = work_df[date_col].dt.to_period("M").astype(str)
                group_col = "_period"
            else:
                work_df["_period"] = "All"
                group_col = "_period"
        except Exception:
            work_df["_period"] = "All"
            group_col = "_period"
    else:
        work_df["_period"] = "All"
        group_col = "_period"

    kpi_frames = []

    # 1. درآمد کل
    if revenue_col and revenue_col in work_df.columns:
        rev_group = work_df.groupby(group_col)[revenue_col].sum().rename("TotalRevenue")
        kpi_frames.append(rev_group)
        kpi_definitions["TotalRevenue"] = {
            "formula": f"SUM({revenue_col})",
            "description": "مجموع درآمد در هر دوره",
            "target": "روند صعودی مطلوب است",
        }

        # 2. تعداد سفارشات و AOV
        if order_col and order_col in work_df.columns:
            orders_count = work_df.groupby(group_col)[order_col].nunique().rename("NumOrders")
            kpi_frames.append(orders_count)
            aov = (rev_group / orders_count).rename("AOV")
            kpi_frames.append(aov)
            kpi_definitions["AOV"] = {
                "formula": f"SUM({revenue_col}) / COUNT(DISTINCT {order_col})",
                "description": "میانگین ارزش هر سفارش (Average Order Value)",
                "target": "هرچه بالاتر بهتر",
            }

        # 3. درآمد به ازای هر مشتری
        if customer_col and customer_col in work_df.columns:
            n_customers = work_df.groupby(group_col)[customer_col].nunique().rename("NumCustomers")
            kpi_frames.append(n_customers)
            rev_per_customer = (rev_group / n_customers).rename("RevenuePerCustomer")
            kpi_frames.append(rev_per_customer)
            kpi_definitions["RevenuePerCustomer"] = {
                "formula": f"SUM({revenue_col}) / COUNT(DISTINCT {customer_col})",
                "description": "میانگین درآمد به ازای هر مشتری",
                "target": "روند صعودی مطلوب است",
            }

    # 4. نرخ بازگشت مشتری
    if customer_col and customer_col in work_df.columns:
        def repeat_rate(sub_df):
            if sub_df.empty:
                return np.nan
            counts = sub_df[customer_col].value_counts()
            if len(counts) == 0:
                return np.nan
            return (counts > 1).sum() / len(counts)

        try:
            rr = work_df.groupby(group_col).apply(repeat_rate, include_groups=False).rename("RepeatRate")
            kpi_frames.append(rr)
            kpi_definitions["RepeatRate"] = {
                "formula": "تعداد مشتریان با بیش از یک تراکنش / کل مشتریان",
                "description": "نرخ بازگشت مشتری در هر دوره",
                "target": "بالای 20-30٪ معمولاً مطلوب است",
            }
        except Exception:
            # در نسخه‌های قدیمی pandas که include_groups را ندارند
            rr = work_df.groupby(group_col).apply(repeat_rate).rename("RepeatRate")
            kpi_frames.append(rr)
            kpi_definitions["RepeatRate"] = {
                "formula": "تعداد مشتریان با بیش از یک تراکنش / کل مشتریان",
                "description": "نرخ بازگشت مشتری در هر دوره",
                "target": "بالای 20-30٪ معمولاً مطلوب است",
            }

    # 5. نرخ داده‌های گمشده
    if not work_df.empty:
        # حذف ستون‌های موقت برای محاسبه missing
        temp_df = work_df.drop(columns=[group_col] if group_col in work_df.columns else [], errors='ignore')
        if not temp_df.empty:
            missing_rate = temp_df.isna().mean(axis=1)
            work_df["_missing_rate"] = missing_rate
            mr = work_df.groupby(group_col)["_missing_rate"].mean().rename("AvgMissingRate")
            kpi_frames.append(mr)
            kpi_definitions["AvgMissingRate"] = {
                "formula": "میانگین درصد مقادیر گمشده در هر سطر",
                "description": "شاخص کیفیت داده در هر دوره",
                "target": "هرچه کمتر بهتر",
            }

    # اگر هیچ KPI ساخته نشد
    if not kpi_frames:
        kpi_df = pd.DataFrame({"Period": ["All"]})
        return kpi_df, {}

    # ترکیب تمام KPIها
    try:
        kpi_df = pd.concat(kpi_frames, axis=1).reset_index()
        kpi_df = kpi_df.rename(columns={group_col: "Period"})
    except Exception as e:
        # اگر خطایی در ترکیب رخ داد، یک دیتافریم خالی برگردان
        print(f"⚠️ خطا در ترکیب KPIها: {e}")
        kpi_df = pd.DataFrame({"Period": ["All"]})
        return kpi_df, {}

    return kpi_df, kpi_definitions


# =========================================================
# 3. تبدیل KPI محاسبه‌شده به متن برای LLM
# =========================================================
def generate_kpi_summary_text(
    kpi_df: pd.DataFrame,
    kpi_definitions: dict,
    period_col: str = "Period",
    max_chars: int = 4000,
) -> str:
    if kpi_df.empty or not kpi_definitions:
        return "## KPIها\nهیچ KPI قابل استخراجی از ستون‌های موجود پیدا نشد.\n"

    lines = ["## تعریف KPIها", ""]
    for name, info in kpi_definitions.items():
        lines.append(
            f"- **{name}**: فرمول = `{info['formula']}` — {info['description']} "
            f"— هدف: {info['target']}"
        )
    lines.append("")

    lines.append("## روند KPIها در طول زمان")
    if period_col in kpi_df.columns:
        try:
            kpi_df = kpi_df.sort_values(period_col)
            lines.append(f"- تعداد دوره‌ها: {len(kpi_df):,}")
        except Exception:
            pass

    numeric_cols = [c for c in kpi_df.select_dtypes(include=[np.number]).columns if c != period_col]

    for col in numeric_cols:
        series = kpi_df[col].dropna()
        if series.empty:
            continue
        
        # محاسبه آمار
        last_val = series.iloc[-1]
        prev_val = series.iloc[-2] if len(series) > 1 else None
        mom = None
        if prev_val is not None and prev_val != 0:
            mom = 100 * (last_val - prev_val) / prev_val

        mean_val = series.mean()
        min_val = series.min()
        max_val = series.max()

        # تشخیص روند با حداقل ۳ نقطه
        if len(series) >= 3:
            try:
                slope = np.polyfit(np.arange(len(series)), series.values, 1)[0]
                if abs(slope) < 0.01 * (abs(mean_val) + 1e-9):
                    trend = "بدون روند مشخص"
                else:
                    trend = "صعودی" if slope > 0 else "نزولی"
            except Exception:
                trend = "قابل تشخیص نیست"
        else:
            trend = "داده کافی برای تشخیص روند نیست"

        # ساخت خط توضیحی
        line = f"- **{col}**: آخرین مقدار = {last_val:.3g}"
        if mom is not None:
            line += f", تغییر نسبت به دوره قبل = {mom:+.1f}%"
        line += f", میانگین = {mean_val:.3g}, min = {min_val:.3g}, max = {max_val:.3g}, روند = {trend}"
        lines.append(line)

    lines.append("")
    result = "\n".join(lines)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n\n(به دلیل محدودیت طول، بخشی از خلاصه KPI حذف شد)"
    return result


# =========================================================
# 4. ارسال به مدل با Retry (برای خطای موقت سرور)
# =========================================================
def _call_model_with_retry(client, model, contents, config, max_retries=5, base_delay=2):
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except genai_errors.ServerError as e:
            if attempt == max_retries - 1:
                raise
            wait_time = base_delay * (2 ** attempt)
            print(f"⚠️ مدل موقتاً در دسترس نیست (تلاش {attempt + 1}/{max_retries})، {wait_time} ثانیه صبر می‌کنم...")
            time.sleep(wait_time)
        except Exception as e:
            # برای سایر خطاها هم تلاش مجدد
            if attempt == max_retries - 1:
                raise
            wait_time = base_delay * (2 ** attempt)
            print(f"⚠️ خطا در ارتباط با مدل (تلاش {attempt + 1}/{max_retries}): {e}")
            time.sleep(wait_time)


# =========================================================
# 5. تابع نهایی: خلاصه + KPI + سوال -> Insight
# =========================================================
def get_llm_insights(
    df: pd.DataFrame,
    data_summary: str | None = None,
    kpi_summary: str | None = None,
    dataset_name: str = "Dataset",
    date_col: str | None = None,
    question: str | None = None,
    model: str = "gemini-2.5-flash",
    fallback_model: str = "gemini-3.0-flash",
    max_tokens: int = 4000,
) -> str:
    """
    پایپ‌لاین کامل AI Excel Analyst.
    اگر data_summary یا kpi_summary از بیرون داده نشود، خودش می‌سازد.
    """
    if data_summary is None:
        data_summary = generate_llm_data_summary(df, dataset_name=dataset_name)

    if kpi_summary is None:
        kpi_df, kpi_definitions = build_auto_kpis(df, date_col=date_col)
        kpi_summary = generate_kpi_summary_text(kpi_df, kpi_definitions)

    if question is None:
        question = """
این دیتاست و KPIهای استخراج‌شده از آن را مانند یک Senior Data Analyst و Business Analyst بررسی کن.

1. برای هر KPI، وضعیت فعلی، روند، و ارزیابی (خوب/نگران‌کننده) را بیان کن.
2. مشکلات کیفیت داده را که ممکن است روی اعتبار KPIها اثر بگذارد مشخص کن.
3. مهم‌ترین Insightهای کسب‌وکاری را بیان کن.
4. برای هر KPI نامطلوب، یک اقدام عملی پیشنهاد بده.
5. اگر KPI خاصی قابل استخراج نبوده، بگو چه ستون‌هایی برای ساخت آن لازم است.
"""

    system_prompt = """
تو یک Senior Data Analyst و Business Analyst باتجربه هستی که در نقش «دستیار هوشمند اکسل» کار می‌کنی.

ورودی تو شامل دو بخش است:
1. خلاصه آماری دیتاست خام (ساختار، کیفیت، آمار ستون‌ها)
2. خلاصه KPIهایی که به‌صورت خودکار از همان دیتاست ساخته شده‌اند (تعریف + روند)

تو هرگز داده خام را نمی‌بینی. بنابراین:
- فقط بر اساس اطلاعات موجود نتیجه‌گیری کن.
- اگر اطلاعات کافی نیست، صریحاً اعلام کن.
- از حدس زدن و ساختن عدد خودداری کن.
- ارتباط بین کیفیت داده (missing values, outlier, ...) و اعتبار KPIها را بررسی کن.

=========================
وظایف تو
=========================

## 1. Executive Summary
در ۳ تا ۵ جمله مهم‌ترین یافته‌ها را خلاصه کن.

## 2. KPI Analysis
برای هر KPI که در خلاصه آمده:
- وضعیت فعلی نسبت به هدف
- روند و تغییر نسبت به دوره قبل
- ارزیابی: خوب / نگران‌کننده / نیاز به بررسی
- علت احتمالی (در صورت وجود شواهد)
- یک اقدام عملی پیشنهادی

## 3. Data Quality Impact
آیا مشکلات کیفیت داده اعتبار KPIها را زیر سوال می‌برد؟

## 4. Business Insights
این داده و KPIها چه داستانی درباره کسب‌وکار تعریف می‌کنند؟ چه تصمیمی می‌توان گرفت؟

## 5. Recommended Next Analyses
حداقل ۵ پیشنهاد عملی.

## 6. Questions for Analyst
حداکثر ۵ سوال مهم برای تحلیل دقیق‌تر.

=========================
قوانین پاسخ
=========================
- پاسخ کاملاً فارسی، با Markdown، Heading و Bullet.
- فقط بر اساس خلاصه‌های داده‌شده پاسخ بده، هیچ چیز را حدس نزن.
"""

    user_prompt = f"""
# Dataset Summary

{data_summary}

# KPI Summary

{kpi_summary}

# User Question

{question}
"""

    # کد جدید و اصلاح‌شده
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("❌ متغیر محیطی GOOGLE_API_KEY یافت نشد. مطمئن شوید در فایل .env مقداردهی شده است.")

# اطمینان از حذف فاصله‌ها و کوتیشن‌های اضافی اطراف API Key
    api_key = api_key.strip().strip("'").strip('"')

# ساخت کلینت با API Key معتبر
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=max_tokens,
    )

    try:
        response = _call_model_with_retry(client, model, user_prompt, config)
    except genai_errors.ServerError:
        print(f"⚠️ مدل {model} در دسترس نیست، تلاش با {fallback_model}...")
        response = _call_model_with_retry(client, fallback_model, user_prompt, config)

    return response.text