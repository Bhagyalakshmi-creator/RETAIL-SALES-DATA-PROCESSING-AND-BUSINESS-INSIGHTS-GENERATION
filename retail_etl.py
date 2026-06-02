import pandas as pd
import numpy as np
import logging
import os

# ==========================================
# CREATE FOLDERS IF NOT EXIST
# ==========================================

os.makedirs("output", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# ==========================================
# LOGGING CONFIGURATION
# ==========================================

logging.basicConfig(
    filename="logs/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Pipeline Started")

try:

    # ==========================================
    # 1. DATA INGESTION
    # ==========================================

    retail1 = pd.read_csv("data/retail_data1.csv")
    retail2 = pd.read_csv("data/retail_data2.csv")
    products = pd.read_csv("data/product_details.csv")

    logging.info("All datasets loaded successfully")

    total_before = len(retail1) + len(retail2)

    # ==========================================
    # 2. COMBINE DATASETS
    # ==========================================

    retail = pd.concat([retail1, retail2], ignore_index=True)

    # ==========================================
    # 3. REMOVE DUPLICATES
    # ==========================================

    before_duplicates = len(retail)

    retail.drop_duplicates(
        subset=["transaction_id"],
        inplace=True
    )

    duplicates_removed = before_duplicates - len(retail)

    # ==========================================
    # 4. HANDLE MISSING VALUES
    # ==========================================

    missing_before = retail.isnull().sum().sum()

    retail["discount"] = retail["discount"].fillna(0)

    retail["city"] = retail["city"].fillna("Unknown")

    retail["purchase_location"] = retail[
        "purchase_location"
    ].fillna("Unknown")

    retail["payment_status"] = retail[
        "payment_status"
    ].fillna("Pending")

    missing_after = retail.isnull().sum().sum()

    fixed_missing = missing_before - missing_after

    # ==========================================
    # 5. STANDARDIZE PRODUCT NAMES
    # ==========================================

    retail["product_name"] = (
        retail["product_name"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    # ==========================================
    # 6. STANDARDIZE CATEGORY
    # ==========================================

    retail["category"] = (
        retail["category"]
        .astype(str)
        .str.strip()
        .str.title()
    )

    # ==========================================
    # 7. REMOVE INVALID QUANTITIES
    # ==========================================

    invalid_qty = len(
        retail[retail["quantity"] <= 0]
    )

    retail = retail[
        retail["quantity"] > 0
    ]

    # ==========================================
    # 8. STANDARDIZE DATES
    # ==========================================

    retail["transaction_date"] = pd.to_datetime(
        retail["transaction_date"],
        errors="coerce"
    )

    invalid_dates = retail[
        retail["transaction_date"].isna()
    ].shape[0]

    retail = retail[
        retail["transaction_date"].notna()
    ]

    # ==========================================
    # 9. MASK EMAIL
    # ==========================================

    def mask_email(email):

        if pd.isna(email):
            return ""

        try:
            name, domain = str(email).split("@")

            return (
                name[:3]
                + "*****@"
                + domain
            )

        except:
            return email

    retail["email"] = retail[
        "email"
    ].apply(mask_email)

    # ==========================================
    # 10. MASK PHONE
    # ==========================================

    def mask_phone(phone):

        phone = str(phone)

        if len(phone) >= 4:

            return (
                phone[:2]
                + "******"
                + phone[-2:]
            )

        return phone

    retail["phone"] = retail[
        "phone"
    ].apply(mask_phone)

    # ==========================================
    # 11. PRODUCT ENRICHMENT
    # ==========================================

    products.columns = (
        products.columns
        .str.strip()
        .str.lower()
    )

    retail.columns = (
        retail.columns
        .str.strip()
        .str.lower()
    )

    if "price_y" in retail.columns:
        retail.drop(
            columns=["price_y"],
            inplace=True
        )

    # ==========================================
    # 12. BUSINESS CALCULATIONS
    # ==========================================

    retail["net_sales"] = (
        retail["price"]
        * retail["quantity"]
    ) - retail["discount"]

    retail["month"] = (
        retail["transaction_date"]
        .dt.strftime("%Y-%m")
    )

    # ==========================================
    # KPI 1 TOTAL REVENUE
    # ==========================================

    total_revenue = pd.DataFrame({
        "Metric": ["Total Revenue"],
        "Value": [retail["net_sales"].sum()]
    })

    # ==========================================
    # KPI 2 CITY SALES
    # ==========================================

    city_sales = (
        retail.groupby("city")
        ["net_sales"]
        .sum()
        .reset_index()
        .sort_values(
            by="net_sales",
            ascending=False
        )
    )

    # ==========================================
    # KPI 3 CATEGORY SALES
    # ==========================================

    category_sales = (
        retail.groupby("category")
        ["net_sales"]
        .sum()
        .reset_index()
        .sort_values(
            by="net_sales",
            ascending=False
        )
    )

    # ==========================================
    # KPI 4 TOP PRODUCTS
    # ==========================================

    top_products = (
        retail.groupby("product_name")
        ["quantity"]
        .sum()
        .reset_index()
        .sort_values(
            by="quantity",
            ascending=False
        )
    )

    # ==========================================
    # KPI 5 MONTHLY SALES
    # ==========================================

    monthly_sales = (
        retail.groupby("month")
        ["net_sales"]
        .sum()
        .reset_index()
    )

    # ==========================================
    # KPI 6 PAYMENT ANALYSIS
    # ==========================================

    payment_analysis = (
        retail.groupby("payment_method")
        ["net_sales"]
        .sum()
        .reset_index()
    )

    # ==========================================
    # KPI 7 CUSTOMER LTV
    # ==========================================

    customer_ltv = (
        retail.groupby("customer_id")
        ["net_sales"]
        .sum()
        .reset_index()
        .sort_values(
            by="net_sales",
            ascending=False
        )
    )

    # ==========================================
    # DATA QUALITY REPORT
    # ==========================================

    quality = pd.DataFrame({

        "Metric": [

            "Original Records",
            "Duplicates Removed",
            "Missing Values Fixed",
            "Invalid Quantity Removed",
            "Invalid Dates Removed",
            "Final Records"

        ],

        "Value": [

            total_before,
            duplicates_removed,
            fixed_missing,
            invalid_qty,
            invalid_dates,
            len(retail)

        ]

    })

    # ==========================================
    # SAVE OUTPUTS
    # ==========================================

    retail.to_csv(
        "output/cleaned_transactions.csv",
        index=False
    )

    city_sales.to_csv(
        "output/city_sales.csv",
        index=False
    )

    category_sales.to_csv(
        "output/category_sales.csv",
        index=False
    )

    top_products.to_csv(
        "output/top_products.csv",
        index=False
    )

    monthly_sales.to_csv(
        "output/monthly_sales.csv",
        index=False
    )

    payment_analysis.to_csv(
        "output/payment_analysis.csv",
        index=False
    )

    customer_ltv.to_csv(
        "output/customer_ltv.csv",
        index=False
    )

    total_revenue.to_csv(
        "output/total_revenue.csv",
        index=False
    )

    quality.to_csv(
        "output/data_quality.csv",
        index=False
    )

    logging.info(
        "All KPI files generated successfully"
    )

    print("ETL Pipeline Executed Successfully!")

except Exception as e:

    logging.error(str(e))

    print("Error:", e)

finally:

    logging.info("Pipeline Completed")