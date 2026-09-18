#!/usr/bin/env python3
"""Collect COCO metrics at epochs 5 and 10 from MMEngine work dirs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


METRIC_ALIASES = {
    'coco/AP': ('coco/AP',),
    'AP50': ('coco/AP .5', 'coco/AP50', 'AP50'),
    'AP75': ('coco/AP .75', 'coco/AP75', 'AP75'),
    'APM': ('coco/AP (M)', 'coco/APM', 'coco/APm', 'APM'),
    'APL': ('coco/AP (L)', 'coco/APL', 'coco/APl', 'APL'),
    'AR': ('coco/AR', 'AR'),
}
TARGET_EPOCHS = (5, 10)


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _metrics_from_mapping(record: dict[str, Any]) -> dict[str, float]:
    result = {}
    for output_name, aliases in METRIC_ALIASES.items():
        for alias in aliases:
            value = _number(record.get(alias))
            if value is not None:
                result[output_name] = value
                break
    return result


def _epoch_from_mapping(record: dict[str, Any]) -> int | None:
    for key in ('epoch', 'Epoch'):
        value = _number(record.get(key))
        if value is not None:
            return int(value)
    # MMEngine's vis_data/scalars.json uses validation epoch as `step`.
    if _metrics_from_mapping(record):
        value = _number(record.get('step'))
        if value is not None:
            return int(value)
    return None


def _parse_json_lines(path: Path) -> list[tuple[int, dict[str, float]]]:
    found = []
    with path.open('r', encoding='utf-8', errors='replace') as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(record, dict):
                continue
            metrics = _metrics_from_mapping(record)
            epoch = _epoch_from_mapping(record)
            if epoch in TARGET_EPOCHS and metrics:
                found.append((epoch, metrics))
    return found


def _parse_text_log(path: Path) -> list[tuple[int, dict[str, float]]]:
    found = []
    epoch_pattern = re.compile(r'Epoch(?:\(val\))?\s*\[\s*(\d+)')
    with path.open('r', encoding='utf-8', errors='replace') as stream:
        for line in stream:
            match = epoch_pattern.search(line)
            if not match:
                continue
            epoch = int(match.group(1))
            if epoch not in TARGET_EPOCHS:
                continue
            metrics = {}
            for output_name, aliases in METRIC_ALIASES.items():
                for alias in aliases:
                    metric_match = re.search(
                        rf'(?<![\w/]){re.escape(alias)}\s*[:=]\s*'
                        r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
                    if metric_match:
                        metrics[output_name] = float(metric_match.group(1))
                        break
            if metrics:
                found.append((epoch, metrics))
    return found


def collect(work_dir: Path) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    candidates = sorted(
        (path for path in work_dir.rglob('*') if path.is_file()
         and (path.suffix in {'.json', '.jsonl', '.log'}
              or path.name.endswith('.log.json'))),
        key=lambda path: (path.stat().st_mtime_ns, str(path)))
    for path in candidates:
        entries = (_parse_json_lines(path) if 'json' in path.suffixes
                   or path.suffix == '.json' else _parse_text_log(path))
        for epoch, metrics in entries:
            row = results.setdefault(epoch, {'source': str(path)})
            row.update(metrics)
            row['source'] = str(path)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--baseline-dir', type=Path,
        default=Path('work_dirs/server_baseline_10e'))
    parser.add_argument(
        '--refine-dir', type=Path,
        default=Path('work_dirs/server_refine_10e'))
    parser.add_argument('--output', type=Path,
                        default=Path('rtmpose_10e_comparison.csv'))
    args = parser.parse_args()

    rows = []
    for model_name, work_dir in (
            ('baseline', args.baseline_dir), ('refine_v1', args.refine_dir)):
        if not work_dir.is_dir():
            print(f'WARNING: work directory not found: {work_dir}')
            values = {}
        else:
            values = collect(work_dir)
        for epoch in TARGET_EPOCHS:
            record = values.get(epoch, {})
            rows.append({
                'model': model_name,
                'epoch': epoch,
                **{name: record.get(name, '') for name in METRIC_ALIASES},
                'source': record.get('source', ''),
            })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['model', 'epoch', *METRIC_ALIASES, 'source']
    with args.output.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {args.output.resolve()}')


if __name__ == '__main__':
    main()

