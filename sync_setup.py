import re
import os

def sync_setup_with_requirements():
    """
    Reads packages from requirements.txt and updates the
    install_requires list in setup.py.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    requirements_path = os.path.join(project_root, 'requirements.txt')
    setup_path = os.path.join(project_root, 'setup.py')

    print("Starting synchronization...")

    # --- 1. Read packages from requirements.txt ---
    try:
        with open(requirements_path, 'r') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        print(f"Found {len(requirements)} packages in {requirements_path}")
    except FileNotFoundError:
        print(f"Error: {requirements_path} not found.")
        return

    # --- 2. Read the content of setup.py ---
    try:
        with open(setup_path, 'r') as f:
            setup_content = f.read()
    except FileNotFoundError:
        print(f"Error: {setup_path} not found.")
        return

    # --- 3. Format the requirements for the install_requires list ---
    # This creates a multi-line string with proper indentation.
    install_requires_str = ",\n        ".join([f'"{req}"' for req in requirements])
    replacement_str = f"install_requires=[\n        {install_requires_str}\n    ]"

    # --- 4. Use regex to replace the existing install_requires list ---
    # This pattern looks for 'install_requires=[' and matches everything
    # until the closing ']' bracket, across multiple lines.
    new_setup_content, num_replacements = re.subn(
        r'install_requires=\[.*?\]',
        replacement_str,
        setup_content,
        flags=re.DOTALL
    )

    # --- 5. Write the updated content back to setup.py ---
    if num_replacements > 0:
        with open(setup_path, 'w') as f:
            f.write(new_setup_content)
        print(f"Successfully synchronized {setup_path} with {requirements_path}.")
    else:
        print("Warning: 'install_requires' list not found in setup.py. No changes made.")

if __name__ == "__main__":
    sync_setup_with_requirements()