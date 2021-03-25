def multicolor_label(ax, list_of_strings, list_of_colors, axis="x", anchorpad=0, **kw):
    """this function creates axes labels with multiple colors
    ax specifies the axes object where the labels should be drawn
    list_of_strings is a list of all of the text items
    list_if_colors is a corresponding list of colors for the strings
    axis='x', 'y', or 'both' and specifies which label(s) should be drawn"""
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

    # x-axis label
    if axis == "x" or axis == "both":
        boxes = [
            TextArea(text, textprops=dict(color=color, ha="left", va="bottom", **kw))
            for text, color in zip(list_of_strings, list_of_colors)
        ]
        xbox = HPacker(children=boxes, align="bottom", pad=0, sep=20)
        anchored_xbox = AnchoredOffsetbox(
            loc="center",
            child=xbox,
            pad=anchorpad,
            frameon=False,
            bbox_to_anchor=(0.5, -0.12),
            bbox_transform=ax.transAxes,
            borderpad=0.0,
        )
        ax.add_artist(anchored_xbox)

    # y-axis label
    if axis == "y" or axis == "both":
        boxes = [
            TextArea(
                text,
                textprops=dict(color=color, ha="left", va="bottom", rotation=90, **kw),
            )
            for text, color in zip(list_of_strings[::-1], list_of_colors)
        ]
        ybox = VPacker(children=boxes, align="center", pad=0, sep=20)
        anchored_ybox = AnchoredOffsetbox(
            loc=3,
            child=ybox,
            pad=anchorpad,
            frameon=False,
            bbox_to_anchor=(-0.12, 0.5),
            bbox_transform=ax.transAxes,
            borderpad=0.0,
        )
        ax.add_artist(anchored_ybox)
