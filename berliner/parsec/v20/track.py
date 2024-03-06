from astropy.table import Table
from collections import OrderedDict


def load_track(f) -> Table:
    with open(f, "r") as f:
        slist = f.readlines()
    idx_begin = -1
    idx_end = -1
    for i, s in enumerate(slist):
        if s.__contains__("BEGIN"):
            idx_begin = i
        if s.__contains__("END"):
            idx_end = i
    if idx_begin < 0:
        raise ValueError("BEGIN not found!")
    if idx_end < 0:
        raise ValueError("END not found!")
    meta = OrderedDict(
        idx_begin=idx_begin,
        idx_end=idx_end,
        n_rec=idx_end - idx_begin - 2,
        header=slist[0].strip(),
    )
    tbl = Table.read(
        slist[idx_begin + 1 : idx_end], format="ascii.basic", fast_reader=True
    )
    tbl.meta = meta
    return tbl


if __name__ == "__main__":
    test_track = (
        "/Volumes/Helium/data/stellar_evolutionary_model/parsec_v2.0/VAR_UNZIPPED/"
        "VAR_ROT0.00_SH_Z0.01_Y0.267/Z0.01Y0.267O_IN0.00OUTA1.74_F7_M.60.TAB.HB"
    )
    t = load_track(test_track)
    t.pprint()
    print(t.meta)
