###############################################################################
# Global Imports
###############################################################################
import csv
from dataclasses import asdict, dataclass
import os
from typing import Optional

###############################################################################
# 3PP Imports
###############################################################################
import click


###############################################################################
# Constants
###############################################################################
HEADERS = [
    "class_id",
    "class_name",
    "filename",
    "original_dataset_uri",
    "platform",
    "mic",
    "channel_orientation",
    "is_phone",
    "clip_id",
    "file_id",
    "day",
    "part",
    "bullet",
    "event_id",
    "mic_location",
    "cartridge",
    "is_silenced",
]


###############################################################################
# Utility Classes
###############################################################################
@dataclass(kw_only=True)
class RawRecord:
    class_id: int
    class_name: str
    filename: str
    original_dataset_uri: str
    platform: Optional[str]
    mic: str
    channel_orientation: Optional[str]
    is_phone: bool
    clip_id: int
    file_id: str
    day: Optional[int]
    part: Optional[int]
    bullet: Optional[str]


@dataclass(kw_only=True)
class ProcRecord(RawRecord):
    # platform to be replaced
    # mic to be replaced
    event_id: str
    mic_location: str
    cartridge: Optional[str]
    is_silenced: bool


###############################################################################
# Helper functions
###############################################################################
def _load_metadata(fq_filename: str) -> list[RawRecord]:
    ret: list[RawRecord] = []
    with open(fq_filename, "r") as infile:
        reader = csv.DictReader(infile)
        for record in reader:
            entry = RawRecord(
                class_id=int(record["class_id"]),
                class_name=record["class_name"],
                filename=record["filename"],
                original_dataset_uri=record["original_dataset_uri"],
                platform=record["platform"],
                mic=record["mic"],
                channel_orientation=record["channel_orientation"],
                is_phone=bool(int(record["is_phone"])),
                clip_id=int(record["clip_id"]),
                file_id=record["file_id"],
                day=int(record["day"]) if record["day"] else None,
                part=int(record["part"]) if record["part"] else None,
                bullet=record["bullet"],
            )
            ret.append(entry)
    return ret


def _uri_to_event_id(dataset_uri: str) -> str:
    if dataset_uri.endswith("CertusGun"):
        return "oh_farm"
    elif "Pete" in dataset_uri:
        return "ny_gravel_pit"
    elif "Sean" in dataset_uri:
        return "nj_farm"
    else:
        raise ValueError(dataset_uri)


def _update_platform(caliber: str, event_id: str, platform: Optional[str]) -> str:
    if event_id == "oh_farm":
        if platform is None:
            raise ValueError(event_id)

        opt = {
            "BENELLIM4": "benelli_m4",
            "WINCHESTER1400": "winchester_1400",
            "CZ457": "czusa_457",
            "GMODEL60": "marlin_gfm60",
            "S&W22": "sw_mp22",
            "SMODEL87A": "springfield_87A",
            "AR15": "custom_ar15_300BLK",
            "WALTHERPPKS": "walther_ppk",
            "SAXDM": "springfieldd_xd",
            "MARLIN1895": "marlin_1895",
            "HKP30L": "hk_P30L",
            "SIGP320": "sig_P320",
            "STEYRC9A2": "steyr_C9A2",
            "HK45": "hk_45",
            "HKMR556A4": "hk_MR556",
            "SPIKESTAC556": "spike_m4_556",
            "SKS": "norinco_sks",
            "LMTMWSAR10": "lmt_ar10",
            "MAUSER": "7mm_FIXME",
            "HKP2000": "hk_P2000",
        }
        return opt[platform]
    elif event_id == "ny_gravel_pit":
        opt = {
            ".380 ACP": "ruger_lcp_380ACP",
            "12 Ga": "remington_870_12ga",
            "16 Ga": "stevens_770",
            "20 Ga": "nef_pardner_20ga",
            ".223 Remington": "savage_223",
            "5.56x45mm": "sw_mp15",
        }
        return opt[caliber]
    else:
        if caliber == "6.5 Creedmoor":
            return "savage_mh110"
        else:
            return "9mm_FIXME"


def _get_cartridge(caliber: str, event_id: str, bullet: Optional[str]) -> Optional[str]:
    if event_id == "oh_farm":
        if caliber == "9x19mm":
            if bullet is None:
                raise ValueError(caliber)
            opt = {
                "115GR": None,
                "124GRHP": "remington_gs_9mm_124",
                "147GR": "blazer_9mm_147",
            }
        else:
            opt = {
                ".300 AAC Blackout": "aac_vmax_300BLK_110",
                ".380 ACP": "blazer_380ACP_95",
                ".22 LR": "cci_mm_22LR_36",
                ".40 S&W": "federal_syn_40SW_205",
                "12 Ga": "federal_tg_12g_8s",
                "45-70": "federal_ps_4570_300",
                "7.62x39mm": "federal_ae_76239_124",
                "20 Ga": "federal_tb_20g_rshp",
                "7.62x51mm": "hsm_762NATO_150",
                "7x57mm Mauser": "ppu_rl_757m_139",
            }

        return opt.get(caliber, None)
    elif event_id == "ny_gravel_pit":
        opt = {
            ".223 Remington": "hornady_08255_223rem_55",
            "16 Ga": "remington_gl_16g_75s",
            ".380 ACP": "remington_umv_380ACP_XX",
            "20 Ga": "winchester_sx_20g_rshp",
            "5.56x45mm": "winchester_tp_556NATO_55",
        }
        return opt.get(caliber, None)
    else:
        if caliber == "6.5 Creedmoor":
            return "sb_ta_65CD_140"
        else:   # 9x19mm
            return "federal_ae_9mm_115"


def _update_mic(file_id: str, event_id: str, caliber: str) -> tuple[str, str]:
    mic_id = int(file_id.split("-")[1])
    if event_id == "oh_farm":
        mics = ["DAVE", "ERIN", "ALICE", "ALICE", "BOB", "BOB", "PIXEL"]
    else:
        mics = ["ALICE", "ALICE", "BOB", "BOB", "CAROL", "CAROL", "DAVE", "ERIN", "BUTCH", "JON"]

    mic = mics[mic_id]
    mic_loc = f"{mic.lower()}_{event_id}"

    note_dist = (event_id == "ny_gravel_pit") and (mic in ["BUTCH", "CAROL", "DAVE"])
    if note_dist:
        dist = "_near" if caliber == "16 Ga" else "_far"
        mic_loc += dist

    return mic, mic_loc


def _make_new_file_id_and_name(record: ProcRecord) -> tuple[str, str]:
    segs = record.file_id.split("-")
    id_portion = "-".join(segs[2:]).replace("-", "0").replace("_", "1")
    filename = f"{record.class_id}-{record.event_id}-{record.platform}-{record.mic}-{id_portion}-{record.clip_id}.wav"
    return id_portion, filename


def _process_record(record: RawRecord) -> ProcRecord:
    caliber = record.class_name
    event_id = _uri_to_event_id(record.original_dataset_uri)

    platform = _update_platform(caliber, event_id, record.platform)
    cartridge = _get_cartridge(caliber, event_id, record.bullet)
    is_silenced = (record.class_name == "7.62x51mm") and (event_id == "oh_farm")

    mic, mic_location = _update_mic(record.file_id, event_id, caliber)

    ret = ProcRecord(
        class_id=record.class_id,
        class_name=record.class_name,
        filename=record.filename,
        original_dataset_uri=record.original_dataset_uri,
        channel_orientation=record.channel_orientation,
        is_phone=record.is_phone,
        clip_id=record.clip_id,
        file_id=record.file_id,
        day=record.day,
        part=record.part,

        bullet=None,
        platform=platform,
        mic=mic,

        event_id=event_id,
        mic_location=mic_location,
        cartridge=cartridge,
        is_silenced=is_silenced,
    )

    file_id, new_filename = _make_new_file_id_and_name(ret)
    ret.file_id = file_id
    ret.filename = new_filename

    return ret


###############################################################################
# ! MAIN
###############################################################################
@click.command()
@click.argument("metadata_filename")
def main(metadata_filename: str):
    meta = _load_metadata(metadata_filename)
    proc_meta = [_process_record(record) for record in meta]

    with open("metadata.csv", "w") as outfile:
        writer = csv.DictWriter(outfile, HEADERS)
        writer.writeheader()
        writer.writerows([asdict(rec) for rec in proc_meta])

    for raw, proc in zip(meta, proc_meta):
        old_file = os.path.join("data", raw.filename)
        new_file = os.path.join("data", proc.filename)
        os.rename(old_file, new_file)


if __name__ == "__main__":
    main()
