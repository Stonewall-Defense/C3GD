# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-08-26

### Added

- ~4000 new data points, including two new classes
- New classes recorded in `metadata/calibers.csv`
- New cartridges recorded in `metadata/cartridges.csv`
- New event recorded in `metadata/events.csv`
- New microphones recorded in `metadata/microphones.csv`
- New microphone locations recorded in `metadata/microphone_locations.csv`
- New platforms recorded in `metadata/platforms.csv`
- Best-effort historical weather conditions recorded in `metadata/weather.csv`
- Full benchmark training script with requirements and parameters

### Changed

- `metadata/events.csv` now notes event start date and weather overview
- Improves `metadata.csv` headers/content to make it easier to match `event_id`, `platform_id`, `cartridge_id`, `mic_id` to match to detailed `metadata/` files

### Fixed

- Audio file sample rate issue
- Dataset name in metadata

## [0.1.0] - 2026-05-14

### Added

- Initial release
