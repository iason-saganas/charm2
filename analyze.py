import urwid
import os

def main_menu_selection(button, index):
    boolean_array_main_menu[0] = boolean_array_main_menu[1] = False  # Set to default
    if index == 0:  # Synthetic inferences
        boolean_array_main_menu[0] = True
        top_synth = open_submenu_of_samples(name_of_mode='synthetic')
        update_widget(top_synth)
    elif index == 1:  # Real inferences
        boolean_array_main_menu[1] = True
        top_real = open_submenu_of_samples(name_of_mode='real')
        update_widget(top_real)


def submenu_analyze_selection(button, index):
    if index == 0:  # Analyze A
        # handle_choice('Compare ground truth with reconstruction signal space')
        raise urwid.ExitMainLoop()
    elif index == 1:  # Go back
        update_widget(main_menu())  # Return to the main menu


def handle_choice(choice):
    # Placeholder for handling specific choices
    # You can add logic for each choice here
    # For now, just display the choice
    update_widget(urwid.Text(f"You selected choice {choice}"))


def update_submenu_choices(directory):
    """
    Search for all .pickle files in the specified directory and return their names.
    """
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return []

    pickle_files = [f for f in os.listdir(directory) if f.endswith('.pickle')]
    if not pickle_files:
        print(f"No .pickle files found in {directory}.")
        return []

    return pickle_files


def display_metadata_for_pickle(pickle_file, directory):
    """
    Display specific lines from the metadata text file for a given .pickle file:
    - The first five lines.
    - Lines 18 through 29.
    """
    timestamp = pickle_file.replace('.pickle', '').replace(' ', '_').replace(':', '-')
    metadata_file = f"metadata_{timestamp}.txt"
    metadata_path = os.path.join(directory, metadata_file)

    if not os.path.exists(metadata_path):
        return f"No metadata file found for {pickle_file}. Searched for: {metadata_path}"

    try:
        with open(metadata_path, 'r') as file:
            lines = file.readlines()

        first_five_lines = lines[:5]  # First five lines
        lines_18_to_29 = lines[17:29]  # Lines 18 to 29 (0-based index)

        display_content = ''.join(first_five_lines)  # Join the first five lines
        display_content += '\n'  # Add a newline between sections
        display_content += '---\n'  # Separator line
        display_content += ''.join(lines_18_to_29)  # Join lines 18 to 29

        return display_content

    except Exception as e:
        return f"An error occurred while reading the file: {e}"


def create_scrollable_metadata_view(metadata_content, associated_pickle_file, name_of_mode):
    """
    Create a scrollable view of the metadata content with options at the top.
    """
    # Add options at the top of the metadata content
    options = ['Analyze this file (see metadata below)', 'Go back to main menu']

    def on_option_selected(button, index):
        if index == 0:  # Analyze
            if name_of_mode == 'synthetic':
                # synthetic mode
                choice_array_synthetic_menu[0] = associated_pickle_file
                update_widget(open_submenu_analyze_synthetic())
            else:
                # real mode
                choice_array_real_menu[0] = associated_pickle_file
                update_widget(open_submenu_analyze_real())
        elif index == 1:  # Go back
            update_widget(main_menu())  # Return to the main menu

    option_buttons = [urwid.Button(option) for option in options]
    for i, button in enumerate(option_buttons):
        urwid.connect_signal(button, 'click', on_option_selected, i)

    option_buttons = [urwid.AttrMap(btn, None, focus_map='reversed') for btn in option_buttons]

    metadata_lines = [urwid.Text(line) for line in metadata_content.split('\n') if line]

    # Combine options and metadata content
    body = option_buttons + [urwid.Divider()] + metadata_lines

    # Create a ListBox to make it scrollable
    listbox = urwid.ListBox(urwid.SimpleFocusListWalker(body))
    padded_listbox = urwid.Padding(listbox, left=2, right=2)
    top = urwid.Frame(padded_listbox)

    return top


def open_submenu_of_samples(name_of_mode, focus_index=0):
    # mode is either `real` or `synthetic`
    directory = 'data_storage/pickled_inferences/' + name_of_mode
    pickle_files = update_submenu_choices(directory)

    if not pickle_files:
        return urwid.Text("No synthetic inferences found.")

    body = [urwid.Text(f"{name_of_mode} inference stored samples:"), urwid.Divider()]

    def show_metadata(button, index):
        metadata_content = display_metadata_for_pickle(pickle_files[index], directory)
        metadata_widget = create_scrollable_metadata_view(metadata_content, associated_pickle_file=pickle_files[index],
                                                          name_of_mode=name_of_mode)
        update_widget(metadata_widget)

    for i, file_name in enumerate(pickle_files):
        button = urwid.Button(file_name)
        urwid.connect_signal(button, 'click', show_metadata, i)
        body.append(urwid.AttrMap(button, None, focus_map='reversed'))

    listbox = urwid.ListBox(urwid.SimpleFocusListWalker(body))
    top = urwid.Padding(listbox, left=2, right=2)
    top = urwid.Frame(top)

    return top


def open_submenu_analyze_synthetic(focus_index=0):
    # Create a dictionary for choices where each value is a dictionary with a 'label' key
    choices = {
        0: {'label': 'Compare ground truth with reconstruction signal space'},
        1: {'label': 'Go back to main menu'}
    }

    # Use create_menu to create the menu
    listbox = create_menu('Select an analysis option:', choices, submenu_analyze_selection, focus_index)

    # Wrap the ListBox in a Padding widget for spacing and Frame for additional UI elements
    top = urwid.Padding(listbox, left=2, right=2)
    top = urwid.Frame(top, footer=urwid.Text('Press q to quit'))

    return top


def toggle_handle(button, index):
    """
    Toggle the state of a boolean handle.
    """

    # Toggle the boolean value
    choice_array_real_menu[index + 1] = not choice_array_real_menu[index + 1]

    # Update button label based on the new state
    button_label = f"[{'x' if choice_array_real_menu[index + 1] else ' '}] Handle {index + 1}"
    button.set_label(button_label)


def open_submenu_analyze_real(focus_index=0):
    # Define choices with placeholders
    choices = {
        0: {'label': 'Handle A'},
        1: {'label': 'Handle B'},
        2: {'label': 'Handle C'},
        3: {'label': 'Handle D'},
        4: {'label': 'Execute'}
    }

    # Create buttons for each choice
    body = [urwid.Text('Select analysis options:')]
    for idx, choice in choices.items():
        if idx < 4: # Only add toggle functionality to the first four handles
            button_label = f"[{'x' if choice_array_real_menu[idx + 1] else ' '}] {choice['label']}"
            button = urwid.Button(button_label)
            urwid.connect_signal(button, 'click', toggle_handle, idx)
        else:  # Execute button
            button_label = f" {choice['label']}"
            button = urwid.Button(button_label)
            urwid.connect_signal(button, 'click', lambda btn: handle_input("q"))
        body.append(urwid.AttrMap(button, None, focus_map='reversed'))

    # Create a ListBox and wrap it in a Frame
    listbox = urwid.ListBox(urwid.SimpleFocusListWalker(body))
    listbox.set_focus(focus_index)
    top = urwid.Padding(listbox, left=2, right=2)
    top = urwid.Frame(top, footer=urwid.Text('Press q to quit'))

    return top


def main_menu():
    choices = ['Synthetic inferences', 'Real inferences']
    menu_choices = {i: {'label': choice} for i, choice in enumerate(choices)}
    return urwid.Padding(create_menu('Here you can analyze stored posterior samples. Press q to quit.',
                                     menu_choices, main_menu_selection), left=2, right=2)


def create_menu(title, choices, callback, focus_index=0):
    body = [urwid.Text(title), urwid.Divider()]
    for idx, choice in choices.items():
        button = urwid.Button(choice['label'])
        urwid.connect_signal(button, 'click', lambda btn, index=idx: callback(btn, index))
        body.append(urwid.AttrMap(button, None, focus_map='reversed'))

    listbox = urwid.ListBox(urwid.SimpleFocusListWalker(body))
    listbox.set_focus(focus_index)
    return listbox


def update_widget(widget):
    global main_loop
    main_loop.widget = widget
    main_loop.draw_screen()


def handle_input(key):
    """
    Handle global key input.
    """
    if key in ('q', 'Q'):
        executeScripts[0] = False  # Set this to false, either ways analysis function will be called without parameters
        raise urwid.ExitMainLoop()


def main():

    global executeScripts
    executeScripts = [True]

    global boolean_array_main_menu  # Boolean options for the main menu: Synthetic [0] or real [1]
    boolean_array_main_menu = [False, False]

    global choice_array_synthetic_menu  # Options for the synthetic sub-menu
    # First element [0]: name of chosen pickle to show ground truth-reconstruction in signal space for.
    choice_array_synthetic_menu = [None]

    global choice_array_real_menu  # Options for the real sub-menu
    # First element [0]: name of chosen pickle to show ground truth-reconstruction in signal space for.
    # All else elements: Boolean values to determine which comparisons to show.
    choice_array_real_menu = [None, False, False, False, False]

    global main_loop
    main_loop = urwid.MainLoop(main_menu(), palette=[('reversed', 'standout', '')], unhandled_input=handle_input)
    main_loop.run()

    # This will run after the main loop has ended
    print("Variables chosen: ")
    print("boolean_array_main_menu: ", boolean_array_main_menu)
    print("choice_array_synthetic_menu: ", choice_array_synthetic_menu)
    print("choice_array_real_menu: ", choice_array_real_menu)

    if executeScripts[0]:

        run_synthetic = boolean_array_main_menu[0]
        run_real = boolean_array_main_menu[1]
        if run_synthetic:
            mode = 0
        else:
            mode = 1

        from scripts.test import my_func
        if mode == 0:
            my_func(mode=mode, arguments=choice_array_synthetic_menu)
        else:
            my_func(mode=mode, arguments=choice_array_real_menu)


if __name__ == '__main__':
    main()
