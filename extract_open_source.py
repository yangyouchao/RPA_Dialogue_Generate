"""Reproduce the selected schema topics and first 50 User 1 personas."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
REVISION = "a520ad7f999ca7e6dfdc25fed9f5070bf6f87b42"
URL = f"https://huggingface.co/datasets/google/Synthetic-Persona-Chat/resolve/{REVISION}/data/Synthetic-Persona-Chat_train.csv"
SELECTION = [
    ("Banks_1", "CheckBalance", "查询账户余额", "查询用户银行账户中的金额。"),
    ("Banks_1", "TransferMoney", "银行转账", "将资金从一个银行账户转入另一位用户的账户。"),
    ("Buses_1", "FindBus", "查询城际巴士", "查询指定两个城市之间的巴士行程。"),
    ("Calendar_1", "GetEvents", "查看日程", "获取用户指定日期的全部日历事项。"),
    ("Calendar_1", "AddEvent", "添加日程", "向用户的日历中添加事项。"),
    ("Events_1", "FindEvents", "查找城市活动", "查找指定城市中的活动。"),
    ("Flights_1", "SearchOnewayFlight", "查询单程航班", "查询前往目的地的单程航班。"),
    ("Flights_1", "SearchRoundtripFlights", "查询往返航班", "查询前往目的地的往返航班。"),
    ("Homes_1", "FindApartment", "寻找公寓", "根据城市和卧室数量寻找公寓。"),
    ("Homes_1", "ScheduleVisit", "预约看房", "预约在指定日期查看某处房产。"),
    ("Hotels_1", "SearchHotel", "查找酒店", "查找指定地点的酒店。"),
    ("Hotels_2", "SearchHouse", "查找短租房屋", "查找指定地点的房屋。"),
    ("Media_1", "FindMovies", "查找点播电影", "按类型查找电影，也可指定导演。"),
    ("Movies_1", "BuyMovieTickets", "购买电影票", "购买指定场次的电影票。"),
    ("Music_1", "LookupSong", "搜索歌曲", "搜索一首歌曲。"),
    ("RentalCars_1", "GetCarsAvailable", "查询可租车辆", "根据城市和日期查询可租用的车辆。"),
    ("Restaurants_1", "FindRestaurants", "查找餐馆", "查找指定城市中供应某类菜系的餐馆。"),
    ("Services_1", "FindProvider", "寻找发型师", "按城市搜索发型师，也可指定其他条件。"),
    ("Travel_1", "FindAttractions", "浏览旅游景点", "浏览指定城市中的旅游景点。"),
    ("Weather_1", "GetWeather", "查询天气", "查询指定地点在某个日期的天气。"),
]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def topics():
    raw = (ROOT / "schema.json").read_bytes()
    services = json.loads(raw.decode("utf-8-sig"))
    index = {s["service_name"]: s for s in services}
    result = []
    for number, (service_id, intent_id, name, translation) in enumerate(SELECTION, 1):
        service = index[service_id]
        intent = next(i for i in service["intents"] if i["name"] == intent_id)
        result.append({"id": f"SGD_{number:03d}", "name": name, "mode": "任务型对话意图",
                       "description_en": intent["description"], "description_zh": translation,
                       "service_name": service_id, "intent_name": intent_id,
                       "seeds": [translation],
                       "boundary": "原数据为任务型服务意图。当前无工具的角色聊天只讨论需求、偏好或给定候选；不得声称完成查询、预约、交易或提供实时结果。"})
    document = {"schema_version": "1.0", "id": "sgd_selected", "name": "开源服务对话主题",
                "version": "2026-09-09.v1", "source": "本地 schema.json 的指定 service/intent；中文为本次人工翻译",
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "selection": "人工选择20个不同的服务意图以覆盖多个领域，非随机抽样；未新增原文件没有的意图。",
                "translation_note": "name与description_zh是翻译；boundary为本项目接入说明，不属于原数据。",
                "subtopics": result}
    daily = json.loads((ROOT / "tmp/dailydialog_20/output/topic_dailydialog_20_zh.json").read_text(encoding="utf-8"))
    fields = list(result[0])
    for item in daily["subtopics"]:
        if list(item) != fields or any(type(item[key]) is not type(result[0][key]) for key in fields):
            raise ValueError("DailyDialog topic fields or types do not match service topics")
    combined = result + daily["subtopics"]
    if len({item["id"] for item in combined}) != len(combined):
        raise ValueError("Duplicate topic id in merged catalog")
    document.update(
        name="开源服务与日常闲聊主题", version="2026-09-14.v2",
        source="本地 schema.json 的20个服务意图，加上 DailyDialog train 提炼的20个闲聊主题。",
        selection="前20条为人工选择的服务意图；后20条为既有 DailyDialog 开场样本提炼，均非本次全量随机抽样。",
        translation_note="SGD描述字段为原意图及中文翻译；DailyDialog描述字段为开场原文及中文译文，name、mode、seeds为提炼。boundary均为项目约束。",
        dailydialog_source={key: value for key, value in daily.items() if key != "subtopics"},
        subtopics=combined)
    save(ROOT / "schema/open_source/topic_schema_20_zh.json", document)
    print(f"Saved {len(combined)} topics: {len(result)} service intents and {len(daily['subtopics'])} DailyDialog topics")


def personas(csv_path=None):
    translations = json.loads((ROOT / "profiles/User_profile/open_source/persona_translations_zh.json").read_text(encoding="utf-8"))
    if csv_path:
        stream = Path(csv_path).open(encoding="utf-8-sig", newline="")
    else:
        response = urllib.request.urlopen(URL, timeout=60)
        stream = io.TextIOWrapper(response, encoding="utf-8-sig", newline="")
    records = []
    with stream:
        reader = csv.DictReader(stream)
        column = "user 1 personas"
        if column not in reader.fieldnames:
            raise ValueError(f"Unexpected CSV columns: {reader.fieldnames}")
        for row_index, row in enumerate(reader):
            if row_index == 50:
                break
            original = row[column]
            lines = original.splitlines()
            missing = [line for line in lines if line not in translations]
            if missing:
                raise ValueError(f"Missing persona translation in source row {row_index}: {missing}")
            records.append({"id": f"SPC_{row_index + 1:03d}", "source_row_index": row_index,
                            "persona": "\n".join(translations[line] for line in lines)})
    if len(records) != 50 or any(not r["persona"].strip() for r in records):
        raise ValueError("Expected first 50 nonempty rows; no skipping or replacement allowed")
    save(ROOT / "profiles/User_profile/open_source/User_profile.json", {
        "schema_version": "1.0", "version": REVISION, "format": "raw_persona",
        "source": "google/Synthetic-Persona-Chat", "source_url": URL, "license": "CC-BY-4.0",
        "split": "train", "column": column,
        "selection": "First 50 data rows in the original CSV, zero-based rows 0..49; User 1 only; no shuffle, deduplication, or inferred attributes. Persona sentences translated to Chinese in their original order.",
        "translation": "Manual Chinese translations in persona_translations_zh.json; original English remains in the pinned source CSV.",
        "profiles": records})
    print(f"Extracted {len(records)} personas; {len(set(r['persona'] for r in records))} unique exact texts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["topics", "personas"])
    parser.add_argument("--csv", type=Path, help="Read a local copy instead of downloading")
    args = parser.parse_args()
    topics() if args.action == "topics" else personas(args.csv)
