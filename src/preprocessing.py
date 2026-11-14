import pandas as pd


def malformed_app_mask(df):
    """Nhận diện bản ghi lệch cột đã phát hiện khi khám phá dữ liệu."""
    return (
        df["Category"].eq("1.9")
        | df["Type"].eq("0")
        | df["Genres"].eq("February 11, 2018")
    ).fillna(False)


def clean_app_rows(df_raw):
    """Loại bản ghi lệch cột và bản sao hoàn toàn từ bảng ứng dụng gốc."""
    df = df_raw.copy()
    records = []

    # ---------- 1. Loại bản ghi lệch cột đã xác định ----------
    rows_before = len(df)

    malformed = malformed_app_mask(df)

    df = df.loc[~malformed].copy()

    records.append({
        "Step": "Remove malformed rows",
        "Rows before": rows_before,
        "Removed": rows_before - len(df),
        "Rows after": len(df)
    })

    # ---------- 2. Loại bản sao hoàn toàn trên các cột gốc ----------
    rows_before = len(df)
    df = df.drop_duplicates(keep="first").copy()

    records.append({
        "Step": "Remove exact duplicates",
        "Rows before": rows_before,
        "Removed": rows_before - len(df),
        "Rows after": len(df)
    })

    return df, pd.DataFrame(records)

def convert_app_numeric(df):
    """Chuyển các cột số, giữ Size gốc và phân biệt placeholder."""
    result = df.copy()

    # ---------- 1. Rating và Reviews ----------
    result["Rating"] = pd.to_numeric(
        result["Rating"], errors="raise"
    ).astype("float64")

    result["Reviews"] = pd.to_numeric(
        result["Reviews"], errors="raise"
    ).astype("Int64")

    # ---------- 2. Installs: giá trị vẫn là cận dưới ----------
    result["Installs"] = pd.to_numeric(
        result["Installs"].astype("string")
        .str.replace(",", "", regex=False)
        .str.replace("+", "", regex=False),
        errors="raise"
    ).astype("Int64")

    # ---------- 3. Price ----------
    result["Price"] = pd.to_numeric(
        result["Price"].astype("string")
        .str.replace("$", "", regex=False),
        errors="raise"
    ).astype("float64")

    # ---------- 4. Size: giữ cột gốc, tạo hai cột dẫn xuất ----------
    size_text = result["Size"].astype("string").str.strip()

    size_varies = size_text.eq("Varies with device").fillna(False)
    size_missing = (
        size_text.isna()
        | size_text.eq("").fillna(False)
    )

    size_parts = size_text.str.extract(r"^(\d+(?:\.\d+)?)([Mk])$")

    unexpected = (
        ~size_missing
        & ~size_varies
        & size_parts[0].isna()
    )

    if unexpected.any():
        bad_values = size_text.loc[unexpected].unique().tolist()
        raise ValueError(f"Size chưa xác định cách chuyển đổi: {bad_values}")

    result["Size_MB"] = (
        pd.to_numeric(size_parts[0], errors="raise")
        * size_parts[1].map({"M": 1.0, "k": 1 / 1024})
    ).astype("float64")

    result["Size_varies"] = size_varies.astype("boolean")

    return result

def finalize_app_types(df):
    """Chuyển ngày, suy ra Type thiếu và chuyển các kiểu còn lại."""
    result = df.copy()

    # ---------- 1. Last Updated ----------
    result["Last Updated"] = pd.to_datetime(
        result["Last Updated"],
        format="%B %d, %Y",
        errors="raise"
    ).astype("datetime64[ns]")

    # ---------- 2. Suy ra Type thiếu từ Price ----------
    type_text = result["Type"].astype("string")

    missing_type = (
        type_text.isna()
        | type_text.str.strip().eq("").fillna(False)
    )

    result.loc[
        missing_type & result["Price"].eq(0), "Type"
    ] = "Free"

    result.loc[
        missing_type & result["Price"].gt(0), "Type"
    ] = "Paid"

    # ---------- 3. Các cột phân loại ----------
    categorical_cols = [
        "Category", "Type", "Content Rating", "Genres"
    ]

    for col in categorical_cols:
        result[col] = result[col].astype("category")

    # ---------- 4. Các cột văn bản ----------
    string_cols = ["App", "Size", "Current Ver", "Android Ver"]

    for col in string_cols:
        result[col] = result[col].astype("string")

    return result


def preprocess_apps(df_raw):
    """Tiền xử lý bảng ứng dụng gốc và báo cáo các bước loại dòng."""
    # Làm sạch dòng trước khi chuyển đổi hoặc thêm cột.
    result, cleaning_report = clean_app_rows(df_raw)

    result = convert_app_numeric(result)
    result = finalize_app_types(result)

    # Tạo index liên tục sau khi hoàn tất.
    result = result.reset_index(drop=True)

    return result, cleaning_report
