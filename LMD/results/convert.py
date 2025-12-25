import pandas as pd


def convert_csv_to_submission(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    # cột class (bỏ cột image)
    class_cols = df.columns[1:]

    submission_rows = []

    for _, row in df.iterrows():
        # lấy image id (bỏ path)
        image_path = row["image"]
        image_id = image_path.split("/")[-1] + ".jpg"

        # tìm class có value = 1
        label = None
        for c in class_cols:
            if row[c] == 1 or row[c] == 1.0:
                label = c
                break

        if label is None:
            raise ValueError(f"No positive label found for image {image_path}")

        submission_rows.append({
            "ID": image_id,
            "labels": label
        })

    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(output_csv, index=False)

    print(f"✅ Submission saved to: {output_csv}")
    print(submission_df.head())


if __name__ == "__main__":
    input_csv = "test_predictions.csv"     # CSV gốc
    output_csv = "submission.csv"            # CSV submission

    convert_csv_to_submission(input_csv, output_csv)
