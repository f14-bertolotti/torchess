def pwn_actions():
    """Return a list of all actions."""

    for src_row in range(8):
        for src_col in range(8):
            for dst_row in range(8):
                for dst_col in range(8):
                    for spc in range(7):
                        yield (src_row, src_col, dst_row, dst_col, spc)



