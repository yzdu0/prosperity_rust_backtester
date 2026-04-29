#!/usr/bin/env python3
"""Build a filtered round dataset by keeping only selected products."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
PEBBLES = {
    "PEBBLES_XS",
    "PEBBLES_S",
    "PEBBLES_M",
    "PEBBLES_L",
    "PEBBLES_XL",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read products from round trades files, keep only the selected products, "
            "and write paired filtered trades/prices CSVs into another round folder."
        )
    )
    parser.add_argument(
        "--source-round",
        type=int,
        default=5,
        help="Round number to read from under datasets/ (default: 5).",
    )
    parser.add_argument(
        "--target-round",
        type=int,
        default=6,
        help="Round number to write under datasets/ (default: 6).",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Optional source directory override. Defaults to datasets/round<SOURCE_ROUND>.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        help="Optional target directory override. Defaults to datasets/round<TARGET_ROUND>.",
    )
    parser.add_argument(
        "--products",
        help="Comma-separated products to keep.",
    )
    parser.add_argument(
        "--product",
        action="append",
        default=[],
        help="Repeatable product to keep. Can be used multiple times.",
    )
    parser.add_argument(
        "--list-products",
        action="store_true",
        help="Print the available products discovered from the source trades files and exit.",
    )
    parser.add_argument(
        "--pebbles",
        action="store_true",
        help="Quick shortcut for all PEBBLES products.",
    )
    return parser.parse_args()


def resolve_round_dir(explicit: Path | None, round_number: int) -> Path:
    if explicit is not None:
        return explicit.resolve()
    return REPO_ROOT / "datasets" / f"round{round_number}"


def collect_trade_files(source_dir: Path) -> list[Path]:
    trade_files = sorted(source_dir.glob("trades_*.csv"))
    if not trade_files:
        raise SystemExit(f"no trades_*.csv files found in {source_dir}")
    return trade_files


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_products_from_trades(trade_files: Iterable[Path]) -> set[str]:
    products: set[str] = set()
    for path in trade_files:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=";")
            if reader.fieldnames is None or "symbol" not in reader.fieldnames:
                raise SystemExit(f"missing 'symbol' column in {path}")
            for row in reader:
                symbol = row["symbol"].strip()
                if symbol:
                    products.add(symbol)
    return products


def parse_selected_products(args: argparse.Namespace) -> set[str]:
    selected = {product.strip() for product in args.product if product.strip()}
    if args.products:
        selected.update(
            product.strip()
            for product in args.products.split(",")
            if product.strip()
        )
    if args.pebbles:
        selected.update(PEBBLES)
    return selected


def remap_round_name(file_name: str, source_round: int, target_round: int) -> str:
    source_token = f"round_{source_round}_"
    target_token = f"round_{target_round}_"
    if source_token not in file_name:
        raise SystemExit(
            f"expected {file_name} to contain {source_token!r} so it can be renamed"
        )
    return file_name.replace(source_token, target_token, 1)


def paired_prices_file(trade_file: Path) -> Path:
    return trade_file.with_name(trade_file.name.replace("trades_", "prices_", 1))


def filter_csv(source: Path, target: Path, product_column: str, selected: set[str]) -> int:
    with source.open("r", encoding="utf-8", newline="") as read_handle:
        reader = csv.DictReader(read_handle, delimiter=";")
        if reader.fieldnames is None or product_column not in reader.fieldnames:
            raise SystemExit(f"missing {product_column!r} column in {source}")

        with target.open("w", encoding="utf-8", newline="") as write_handle:
            writer = csv.DictWriter(
                write_handle,
                fieldnames=reader.fieldnames,
                delimiter=";",
                lineterminator="\n",
            )
            writer.writeheader()

            row_count = 0
            for row in reader:
                if row[product_column].strip() in selected:
                    writer.writerow(row)
                    row_count += 1
    return row_count


def main() -> int:
    args = parse_args()
    source_dir = resolve_round_dir(args.source_dir, args.source_round)
    target_dir = resolve_round_dir(args.target_dir, args.target_round)

    if not source_dir.is_dir():
        raise SystemExit(f"source directory does not exist: {source_dir}")

    trade_files = collect_trade_files(source_dir)
    available_products = read_products_from_trades(trade_files)

    if args.list_products:
        for product in sorted(available_products):
            print(product)
        return 0

    selected_products = parse_selected_products(args)
    if not selected_products:
        raise SystemExit(
            "choose at least one product with --products or one or more --product flags"
        )

    unknown_products = sorted(selected_products - available_products)
    if unknown_products:
        raise SystemExit(
            "unknown products: "
            + ", ".join(unknown_products)
            + "\nUse --list-products to inspect the available round products."
        )

    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Keeping {len(selected_products)} products:")
    for product in sorted(selected_products):
        print(f"  {product}")

    for trade_file in trade_files:
        prices_file = paired_prices_file(trade_file)
        if not prices_file.is_file():
            raise SystemExit(f"missing paired prices file for {trade_file.name}")

        target_trade = target_dir / remap_round_name(
            trade_file.name,
            args.source_round,
            args.target_round,
        )
        target_prices = target_dir / remap_round_name(
            prices_file.name,
            args.source_round,
            args.target_round,
        )

        trade_rows = filter_csv(trade_file, target_trade, "symbol", selected_products)
        price_rows = filter_csv(prices_file, target_prices, "product", selected_products)

        print(f"Wrote {display_path(target_trade)} with {trade_rows} rows")
        print(f"Wrote {display_path(target_prices)} with {price_rows} rows")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
