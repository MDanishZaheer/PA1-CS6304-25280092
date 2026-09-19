# Name : Muhammad Danish Zaheer Awan
# REG id : 25280092

"""Compare target-aware Task 2 DAN with target-free Task 3 DAN-DG."""

from pathlib import Path

import pandas as pd


def create_task2_task3_summary(task2_summary_file, task3_summary_table):
    """Create an aligned aggregate table across the two PACS tasks."""
    task2_path = Path(task2_summary_file)
    if not task2_path.exists():
        return None
    task2_summary = pd.read_csv(task2_path)
    required_task2 = task2_summary[
        task2_summary["result_name"].isin(["source_only", "dan"])
    ].copy().reset_index(drop=True)
    required_task3 = task3_summary_table[
        task3_summary_table["result_name"].isin(["erm", "dan_dg", "sam"])
    ].copy().reset_index(drop=True)

    task2_rows = pd.DataFrame(
        {
            "experiment": required_task2["result_name"].map(
                {"source_only": "Shared ERM", "dan": "Task 2 DAN"}
            ),
            "target_access_during_training": required_task2["result_name"].map(
                {"source_only": "No", "dan": "Unlabeled Sketch"}
            ),
            "mean_source_accuracy": required_task2["mean_source_accuracy"],
            "mean_source_macro_f1": required_task2["mean_source_macro_f1"],
            "sketch_accuracy": required_task2["target_accuracy"],
            "sketch_macro_f1": required_task2["target_macro_f1"],
        }
    )
    task3_rows = pd.DataFrame(
        {
            "experiment": required_task3["result_name"].map(
                {
                    "erm": "Shared ERM",
                    "dan_dg": "Task 3 DAN-DG",
                    "sam": "Task 3 SAM",
                }
            ),
            "target_access_during_training": "No",
            "mean_source_accuracy": required_task3["mean_source_accuracy"],
            "mean_source_macro_f1": required_task3["mean_source_macro_f1"],
            "sketch_accuracy": required_task3["sketch_accuracy"],
            "sketch_macro_f1": required_task3["sketch_macro_f1"],
        }
    )
    combined = pd.concat([task2_rows, task3_rows], ignore_index=True)
    return combined.drop_duplicates(subset=["experiment"], keep="first")


def create_task2_task3_class_comparison(
    task2_metrics_directory,
    task3_metrics_directory,
):
    """Compare DAN and DAN-DG per-class changes against the shared ERM."""
    task2_file = Path(task2_metrics_directory) / "dan_class_changes.csv"
    task3_file = Path(task3_metrics_directory) / "dan_dg_class_changes.csv"
    if not task2_file.exists() or not task3_file.exists():
        return None
    task2_table = pd.read_csv(task2_file)[
        ["class_index", "class_name", "accuracy_change"]
    ].rename(columns={"accuracy_change": "task2_dan_accuracy_change"})
    task3_table = pd.read_csv(task3_file)[
        ["class_index", "class_name", "accuracy_change"]
    ].rename(columns={"accuracy_change": "task3_dan_dg_accuracy_change"})
    return task2_table.merge(
        task3_table,
        on=["class_index", "class_name"],
        validate="one_to_one",
    )
