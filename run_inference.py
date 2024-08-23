import urwid


def main_menu_selection(button, index):
    if index == 0:  # Run synthetic inference
        boolean_array_main_menu[index] = True
        update_widget(open_submenu_synthetic_inference())
    elif index == 1:  # Run real inference
        boolean_array_main_menu[index] = True
        raise urwid.ExitMainLoop()  # Exit after selection


def update_submenu_choices():
    choices = [
        f'      [{"x" if boolean_array_synthetic_menu[0] else " "}] Plot ground truth',
        f'      [{"x" if boolean_array_synthetic_menu[1] else " "}] Plot mock data',
        "       ➤  Execute (exit)",
        "       ⮐  Back to main menu (should work)"
    ]
    return choices


def menu(title, choices, callback, focus_index=0):
    body = [urwid.Text(title), urwid.Divider()]
    for i, choice in enumerate(choices):
        button = urwid.Button(choice)
        urwid.connect_signal(button, 'click', lambda btn, idx=i: callback(btn, idx))
        body.append(urwid.AttrMap(button, None, focus_map='reversed'))

    listbox = urwid.ListBox(urwid.SimpleFocusListWalker(body))
    listbox.set_focus(focus_index)  # Set the focus to the desired index
    return listbox


def open_submenu_synthetic_inference(focus_index=0):
    choices = update_submenu_choices()
    listbox = menu('Synthetic Inference Options:', choices, synthetic_submenu_update_bools, focus_index)

    # Wrap the listbox in a padding widget
    top = urwid.Padding(listbox, left=2, right=2)
    top = urwid.Frame(top, footer=urwid.Text('Press ENTER to select'))

    return top


def synthetic_submenu_update_bools(button, index):
    if index == 0:  # Toggle Plot ground truth
        boolean_array_synthetic_menu[0] = not boolean_array_synthetic_menu[0]
    elif index == 1:  # Toggle Plot mock data
        boolean_array_synthetic_menu[1] = not boolean_array_synthetic_menu[1]
    elif index == 2:  # Execute the synthetic inference
        raise urwid.ExitMainLoop()
    elif index == 3:  # Back to main menu
        update_widget(main_menu())
        return

    # Reopen the submenu with the same focus index
    update_widget(open_submenu_synthetic_inference(focus_index=index+2))


def main_menu():
    choices = ['   Run synthetic inference', '   Run real inference (exit)']
    menu_choices = {i: {'label': choice} for i, choice in enumerate(choices)}
    return urwid.Padding(create_menu('Select an option:', menu_choices, main_menu_selection), left=2, right=2)


def create_menu(title, choices, callback, focus_index=0):
    body = [urwid.Text(title), urwid.Divider()]
    for idx, choice in choices.items():
        button = urwid.Button(choice['label'])
        urwid.connect_signal(button, 'click', lambda btn, index=idx: callback(btn, index))
        body.append(urwid.AttrMap(button, None, focus_map='reversed'))

    listbox = urwid.ListBox(urwid.SimpleFocusListWalker(body))
    listbox.set_focus(focus_index)  # Set the focus to the desired index
    return listbox


def update_widget(widget):
    global main_loop
    main_loop.widget = widget
    main_loop.draw_screen()


def main():
    global boolean_array_main_menu  # Boolean options for the main menu
    boolean_array_main_menu = [False, False]

    global boolean_array_synthetic_menu  # Boolean options for the synthetic menu
    boolean_array_synthetic_menu = [False, False]

    global main_loop
    main_loop = urwid.MainLoop(main_menu(), palette=[('reversed', 'standout', '')])
    main_loop.run()

    synthetic_inference = boolean_array_main_menu[0]
    cosmological_inference = boolean_array_main_menu[1]

    plot_mock_ground_truth = boolean_array_synthetic_menu[0]
    plot_mock_data = boolean_array_synthetic_menu[1]

    print("synthetic_inference: ", synthetic_inference, "plot_mock_ground_truth", plot_mock_ground_truth,
          "plot_mock_data: ", plot_mock_data)

    if synthetic_inference:
        from scripts.synthetic_inference import main_synthetic
        main_synthetic(plot_ground_truth=plot_mock_ground_truth, plot_mock_data=plot_mock_data)


if __name__ == '__main__':
    main()
