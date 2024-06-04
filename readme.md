To set up the project, follow these steps:
0. Clone the repository:
    ```bash
    git clone
    ```

1. Copy the `.env.example` file and rename it to `.env` and fill in the required values:
    ```bash
    cp .env.example .env
    ```

2. Create a virtual environment (venv):
    ```bash
    python -m venv myenv
    ```

3. Activate the virtual environment:
    - For Windows:
      ```bash
      myenv\Scripts\activate
      ```
    - For macOS/Linux:
      ```bash
      source myenv/bin/activate
      ```

4. Install the project dependencies from the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

5. Run the `main.py` script:
    ```bash
    python main.py
    ```