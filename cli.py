"""
cli.py

CLI ساده برای AI Excel Analyst.

نحوه استفاده (توی ترمینال):
    python cli.py path/to/file.xlsx
    python cli.py path/to/file.xlsx --sheet Sheet1
    python cli.py path/to/file.xlsx --date-col OrderDate
    python cli.py path/to/file.xlsx --output result.md
    python cli.py path/to/file.xlsx --question "روی مشتریان پرریسک تمرکز کن"
"""

import argparse
from html import parser
import os
import sys
import pandas as pd
from dotenv import load_dotenv

# بارگذاری متغیرهای محیطی از فایل .env
load_dotenv()

from core import (
    generate_llm_data_summary,
    build_auto_kpis,
    generate_kpi_summary_text,
    get_llm_insights,
)


def load_excel(file_path: str, sheet: str | None = None) -> pd.DataFrame:
    """بارگذاری فایل اکسل یا CSV با مدیریت خطا"""
    if not os.path.exists(file_path):
        print(f"❌ فایل پیدا نشد: {file_path}")
        sys.exit(1)

    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path, sheet_name=sheet or 0)
        elif ext == ".csv":
            # تلاش برای تشخیص خودکار encoding
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='utf-8-sig')
        else:
            print(f"❌ فرمت فایل پشتیبانی نمی‌شود: {ext}")
            sys.exit(1)
    except Exception as e:
        print(f"❌ خطا در خواندن فایل: {e}")
        sys.exit(1)

    if df.empty:
        print("❌ فایل خالی است یا داده‌ای در آن یافت نشد.")
        sys.exit(1)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="AI Excel Analyst — تحلیل خودکار فایل اکسل با LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
مثال‌ها:
  python cli.py sales.xlsx
  python cli.py sales.xlsx --sheet Q1 --date-col Date
  python cli.py sales.xlsx --question "تحلیل فروش ماهانه" --output report.md
        """
    )
    parser.add_argument("file", help="مسیر فایل اکسل یا CSV")
    parser.add_argument("--sheet", default=None, help="نام شیت (پیش‌فرض: شیت اول)")
    parser.add_argument("--date-col", default=None, help="نام ستون تاریخ (اختیاری)")
    parser.add_argument("--question", default=None, help="سوال سفارشی برای تحلیلگر (اختیاری)")
    parser.add_argument("--model", default="gemini-3.5-flash", help="نام مدل Gemini")
    parser.add_argument("--output", default=None, help="مسیر فایل خروجی (مثلاً result.md)")
    parser.add_argument("--dataset-name", default=None, help="نامی که در خلاصه نمایش داده می‌شود")
    parser.add_argument("--no-kpi", action="store_true", help="ساخت KPI خودکار را غیرفعال کن")


    
    args = parser.parse_args()

    # بررسی وجود API Key
    if not os.environ.get("GOOGLE_API_KEY"):
        print("❌ متغیر محیطی GOOGLE_API_KEY تنظیم نشده است.")
        print("   برای تنظیم آن از دستور زیر استفاده کنید:")
        print("   export GOOGLE_API_KEY='your-api-key-here'")
        print("   یا یک فایل .env در کنار فایل cli.py قرار دهید با محتوای:")
        print("   GOOGLE_API_KEY=your-api-key-here")
        sys.exit(1)

    dataset_name = args.dataset_name or os.path.basename(args.file)

    print(f"📂 در حال خواندن فایل: {args.file}")
    df = load_excel(args.file, sheet=args.sheet)
    print(f"✅ فایل خوانده شد — {df.shape[0]:,} سطر, {df.shape[1]} ستون")

    print("📊 در حال ساخت خلاصه دیتاست...")
    data_summary = generate_llm_data_summary(df, dataset_name=dataset_name)

    if not args.no_kpi:
        print("📈 در حال ساخت KPIهای خودکار...")
        kpi_df, kpi_definitions = build_auto_kpis(df, date_col=args.date_col)
        kpi_summary = generate_kpi_summary_text(kpi_df, kpi_definitions)

        if not kpi_definitions:
            print("⚠️ هیچ KPI شناخته‌شده‌ای از ستون‌های این فایل ساخته نشد.")
        else:
            print(f"✅ {len(kpi_definitions)} KPI ساخته شد: {', '.join(kpi_definitions.keys())}")
    else:
        print("⏭️ ساخت KPI غیرفعال شد.")
        kpi_summary = "## KPIها\nساخت KPI توسط کاربر غیرفعال شده است."

    print("🤖 در حال گرفتن Insight از مدل...")
    try:
        insights = get_llm_insights(
            df=df,
            data_summary=data_summary,
            kpi_summary=kpi_summary,
            dataset_name=dataset_name,
            date_col=args.date_col,
            question=args.question,
            model=args.model,
        )
    except Exception as e:
        print(f"❌ خطا در دریافت پاسخ از مدل: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("📝 نتیجه تحلیل")
    print("=" * 60 + "\n")
    print(insights)

    if args.output:
        # اطمینان از وجود دایرکتوری خروجی
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(f"# گزارش تحلیل: {dataset_name}\n\n")
            f.write(f"**تاریخ تحلیل:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## خلاصه دیتاست\n\n")
            f.write(data_summary)
            f.write("\n\n## خلاصه KPIها\n\n")
            f.write(kpi_summary)
            f.write("\n\n## Insight نهایی\n\n")
            f.write(insights)
        print(f"\n💾 نتیجه در فایل ذخیره شد: {args.output}")


if __name__ == "__main__":
    main()
    