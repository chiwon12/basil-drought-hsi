#!/usr/bin/env python3
"""
Zenodo: published 레코드에 New version을 만들어 단일 zip을 교체 업로드.

전제: 레코드가 이미 published라 기존 파일을 직접 삭제할 수 없음.
      -> newversion 액션으로 draft를 만들고, draft에 복사돼온 옛 파일을 지운 뒤
         새 zip을 bucket API로 업로드한다. publish는 --publish 줬을 때만.

사용:
  export ZENODO_TOKEN=xxxxx          # zenodo.org > Applications > Personal access tokens
  python zenodo_new_version.py --record 20368943 \
      --zip /mnt/mydisk/chiwon/revision/revision_v2/basil_hsi_subset.zip \
      --version 2.0.0                # 선택: 메타데이터 version 갱신
  # 위까지가 dry(업로드만, 미발행). 내용 확인 후:
  python zenodo_new_version.py --record 20368943 --zip ... --resume --publish
"""
import argparse, os, sys, requests

BASE = "https://zenodo.org/api"   # 테스트는 https://sandbox.zenodo.org/api


def H(token):
    return {"Authorization": f"Bearer {token}"}


def jprint(label, r):
    print(f"[{label}] HTTP {r.status_code}")
    if r.status_code >= 400:
        print(r.text[:1000]); sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True, help="published record/deposition id, e.g. 20368943")
    ap.add_argument("--zip", required=True, help="path to single zip to upload")
    ap.add_argument("--version", help="metadata version string to set on the new draft")
    ap.add_argument("--publish", action="store_true", help="실제 발행까지 진행")
    ap.add_argument("--base", default=BASE)
    args = ap.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("ZENODO_TOKEN 환경변수가 없음")
    if not os.path.isfile(args.zip):
        sys.exit(f"zip 없음: {args.zip}")
    base = args.base
    s = requests.Session(); s.headers.update(H(token))

    # 1) 새 버전 draft 생성 (이미 draft 있으면 그걸 재사용)
    r = s.post(f"{base}/deposit/depositions/{args.record}/actions/newversion")
    jprint("newversion", r)
    latest_draft = r.json()["links"]["latest_draft"]
    draft_id = latest_draft.rstrip("/").split("/")[-1]
    print(f"draft id = {draft_id}")

    # 2) draft 상세 (bucket, 복사돼온 옛 파일들)
    r = s.get(f"{base}/deposit/depositions/{draft_id}"); jprint("get draft", r)
    dep = r.json()
    bucket = dep["links"]["bucket"]
    old_files = dep.get("files", [])
    print(f"bucket = {bucket}")
    print(f"옛 파일 {len(old_files)}개 (삭제 예정): " + ", ".join(f['filename'] for f in old_files))

    # 3) draft에 복사돼온 옛 파일 전부 삭제
    for f in old_files:
        rd = s.delete(f"{base}/deposit/depositions/{draft_id}/files/{f['id']}")
        print(f"  deleted {f['filename']} -> HTTP {rd.status_code}")

    # 4) 새 zip 업로드 (bucket PUT = 대용량/스트리밍)
    fname = os.path.basename(args.zip)
    size = os.path.getsize(args.zip)
    print(f"업로드 시작: {fname} ({size/1e9:.2f} GB) ... 시간 걸림")
    with open(args.zip, "rb") as fp:
        ru = requests.put(f"{bucket}/{fname}", data=fp, headers=H(token))
    jprint("upload", ru)
    print(f"  업로드 완료: {ru.json().get('checksum')}")

    # 5) (선택) version 메타데이터 갱신
    if args.version:
        meta = dep["metadata"]; meta["version"] = args.version
        rm = s.put(f"{base}/deposit/depositions/{draft_id}",
                   json={"metadata": meta})
        jprint("update meta", rm)

    # 6) 발행
    if args.publish:
        rp = s.post(f"{base}/deposit/depositions/{draft_id}/actions/publish")
        jprint("publish", rp)
        print("발행 완료! DOI:", rp.json().get("doi"))
        print("URL:", rp.json()["links"].get("record_html"))
    else:
        print("\n=== 미발행(draft) 상태로 멈춤. 웹에서 확인 후 --publish 로 재실행 ===")
        print("draft 확인:", f"https://zenodo.org/deposit/{draft_id}")


if __name__ == "__main__":
    main()
