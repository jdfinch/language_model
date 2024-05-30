

# Running on a Server Environment

In this tutorial, we'll walk through setting up a remote development environment using VS Code or PyCharm Pro, cloning a project, and configuring it to run on a server with Miniconda. We'll cover steps including creating a virtual environment, automatic deployment, running scripts remotely, and utilizing SLURM for GPU jobs.

There's a lot of steps here which can be overwhelming, but remember you can just ask chatgpt to help you with any of these steps if you run into trouble, most of the steps are commonly talked about on the internet so chatgpt usually knows what to do.

### Prerequisites:
- VS Code or PyCharm Pro installed on your local machine
- SSH access to a remote server
- Basic knowledge of Linux command line

### Steps:

1. **Create a Project Locally:**
   - Open your preferred IDE (VS Code or PyCharm Pro, but Pycharm Pro is recommended and is free for students).
   - Create a new project in your desired location.

2. **Clone Language_Model:**
   - Inside your project, clone the `language_model` repository as a subfolder.

3. **Mark as Sources Root/Python Path:**
   - Mark the `language_model` folder as a sources root or add it to your Python path to enable importing.

4. **Log into the Server:**
   - SSH into your server using your terminal: `ssh username@server_ip`. Remember to use the VPN if required before you try to connect!

5. **Install Miniconda:**
   - Navigate to your desired directory (e.g., `/local/scratch/yourusername`).
   - Download and install Miniconda

6. **Create Virtual Environment:**
   - Create a new virtual environment using Miniconda:
     ```bash
     conda create -n myenv python=3.10
     ```

7. **Create Project Directory on Server:**
   - Create a directory for your project on the server (e.g., `/local/scratch/yourusername/myproject`).

8. **Set Up Remote Interpreter:**
   - In your local IDE, set up a remote interpreter that links to the virtual environment on the server. You will need to link to the virtual env binary like `/local/scratch/username/miniconda3/envs/myenv/bin/python`.

9. **Automatic Deployment:**
    - Configure automatic deployment to link your local project to the empty server folder you created. This ensures any file changes are automatically uploaded to the server.

10. **Verify Script Execution:**
    - Create a new python script. Print 'hello world' in your script and verify you can run it using the remote interpreter.

11. **Install Requirements:**
    - SSH back into the server, activate your virtual environment, and install the requirements of `language_model` using pip:
      ```bash
      conda activate myenv
      pip install -r /local/scratch/username/myproject/language_model/requirements.txt
      ```

12. **Verify Requirements Installed:**
    - Add `import torch` to your script and verify that the remote interpreter can import it successfully. Try `from langauage_model.t5 import T5` to verify importing the language_model package as well.

13. **Create Shell Script:**
    - Create a shell script (e.g., `run.sh`) in your project directory. This script sets necessary environment variables and runs your Python script. Please refer to `docs/run.sh` for an example.

14. **Test Shell Script:**
    - SSH into the server, make the shell script executable, and run it:
      ```bash
      chmod +x run.sh
      ./run.sh
      ```

15. **Run Script Using SLURM:**
    - Run your script using SLURM for GPU jobs:
      ```bash
      sbatch run.sh
      ```

16. **View Job Output:**
    - View the output of the job using:
      ```bash
      cat job.out
      ```
      To view output as it streams, use:
      ```bash
      tail -f job.out
      ```

17. **Coding and Running:**
    - You're now set up to code as usual in your script. Any changes you save will automatically upload to the server. Run your script using `sbatch` on your shell script through SSH.

Congratulations! You've successfully set up a remote development environment for your project, enabling you to code, deploy, and run scripts seamlessly on a remote server.
