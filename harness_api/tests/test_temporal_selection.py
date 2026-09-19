import unittest

from app.v2.raster_grid import NativeSCL
from app.v2.raster_math import CLOUD_POLICY, NativeBand
from app.v2.schemas import SpatialBoundingBox, TemporalExtent
from app.v2.temporal import (
    TemporalInputProfile,
    TemporalSelectAlignArguments,
    coverage_fraction,
    select_temporal_pair,
)


BBOX = [118.79, 31.99, 118.81, 32.01]


def candidate(
    item: str,
    acquired: str,
    cloud: float,
    *,
    bbox=BBOX,
    platform="sentinel-2",
    instrument="msi",
):
    common = dict(
        sha256=(item.encode().hex() + "0" * 64)[:64],
        item_id=item,
        acquired=acquired,
        crs="EPSG:32650",
        transform=[10.0, 0.0, 669060.0, 0.0, -10.0, 3542980.0],
        width=194,
        height=226,
    )
    red = NativeBand(
        asset_id="red-" + item,
        band="red",
        dtype="uint16",
        scale=0.0001,
        offset=-0.1,
        nodata=0.0,
        **common,
    )
    scl = NativeSCL(
        asset_id="scl-" + item,
        band="scl",
        dtype="uint8",
        scale=1.0,
        offset=0.0,
        nodata=0.0,
        transform=[20.0, 0.0, 669060.0, 0.0, -20.0, 3542980.0],
        width=97,
        height=113,
        **{key: value for key, value in common.items() if key not in {"transform", "width", "height"}},
    )
    return TemporalInputProfile(
        item_id=item,
        acquired=acquired,
        platform=platform,
        instrument=instrument,
        red=red,
        scl=scl,
        bbox_wgs84=bbox,
        cloud_fraction=cloud,
    )


def arguments(**updates):
    values = dict(
        operation="select_align",
        before=TemporalExtent(
            start="2024-04-01T00:00:00Z", end="2024-04-10T23:59:59Z"
        ),
        after=TemporalExtent(
            start="2024-05-01T00:00:00Z", end="2024-05-20T23:59:59Z"
        ),
        aoi=SpatialBoundingBox(west=118.79, south=31.99, east=118.81, north=32.01),
        band="red",
        minimum_coverage_fraction=0.95,
        maximum_cloud_fraction=0.01,
        cloud_policy=CLOUD_POLICY,
    )
    values.update(updates)
    return TemporalSelectAlignArguments(**values)


class TemporalSelectionTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            candidate("s2-20240405", "2024-04-05T02:58:52.954000Z", 0.0),
            candidate("s2-20240415", "2024-04-15T02:58:58.108000Z", 0.0),
            candidate("s2-20240510", "2024-05-10T02:58:56.417000Z", 0.00356),
        ]

    def test_selects_valid_pair_deterministically(self):
        result = select_temporal_pair(arguments(), list(reversed(self.candidates)))
        self.assertEqual(result.status, "selected")
        self.assertEqual(result.before.item_id, "s2-20240405")
        self.assertEqual(result.after.item_id, "s2-20240510")
        self.assertEqual(result.before.coverage_fraction, 1.0)
        self.assertEqual(result.considered_item_ids, sorted(result.considered_item_ids))

    def test_rejects_cloudy_window(self):
        result = select_temporal_pair(
            arguments(maximum_cloud_fraction=0.001), self.candidates
        )
        self.assertEqual((result.status, result.reason), ("rejected", "cloudy"))
        self.assertIsNone(result.before)

    def test_rejects_insufficient_coverage(self):
        result = select_temporal_pair(
            arguments(
                aoi=SpatialBoundingBox(
                    west=118.788, south=31.988, east=118.812, north=32.012
                )
            ),
            self.candidates,
        )
        self.assertEqual(result.reason, "insufficient_coverage")

    def test_rejects_wrong_date(self):
        result = select_temporal_pair(
            arguments(
                after=TemporalExtent(
                    start="2024-06-01T00:00:00Z", end="2024-06-20T23:59:59Z"
                )
            ),
            self.candidates,
        )
        self.assertEqual(result.reason, "wrong_date")

    def test_rejects_sensor_mismatch(self):
        values = [
            self.candidates[0],
            candidate(
                "landsat-20240510",
                "2024-05-10T02:58:56.417000Z",
                0.0,
                platform="landsat-9",
                instrument="oli-2",
            ),
        ]
        self.assertEqual(select_temporal_pair(arguments(), values).reason, "sensor_mismatch")

    def test_coverage_uses_requested_aoi_denominator(self):
        aoi = SpatialBoundingBox(
            west=118.78, south=31.98, east=118.82, north=32.02
        )
        value = coverage_fraction(BBOX, aoi)
        self.assertGreater(value, 0.24)
        self.assertLess(value, 0.26)

    def test_rejects_overlapping_windows(self):
        with self.assertRaises(ValueError):
            arguments(
                after=TemporalExtent(
                    start="2024-04-10T00:00:00Z", end="2024-04-20T00:00:00Z"
                )
            )

    def test_rejects_duplicate_candidate_identity(self):
        with self.assertRaisesRegex(ValueError, "item IDs"):
            select_temporal_pair(arguments(), [self.candidates[0], self.candidates[0]])


if __name__ == "__main__":
    unittest.main()
