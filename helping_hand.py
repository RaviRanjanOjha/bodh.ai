import os

def generate_code_report(root_dir, output_file='code_report.md'):
    """
    Creates a markdown file containing all code files with filename tags
    from a directory and its subdirectories.
    
    Args:
        root_dir (str): Path to root directory containing code files
        output_file (str): Name of output markdown file
            """
    # Supported file extensions with their corresponding language tags
    SUPPORTED_EXTENSIONS = {
        '.py': 'python',
        '.js': 'javascript',
        '.jsx': 'jsx',
        '.ts': 'typescript',
        '.tsx': 'tsx',
        '.css': 'css',
        '.scss': 'scss',
        '.html': 'html'
    }
    
    with open(output_file, 'w', encoding='utf-8') as md_file:
        md_file.write("# Code Structure Report\n\n")
        md_file.write("## Directory Structure\n\n")
        md_file.write("```\n")
        md_file.write(generate_directory_tree(root_dir))
        md_file.write("\n```\n\n")
        
        processed_files = 0
        
        for root, dirs, files in os.walk(root_dir):
            # Skip hidden directories, metadata, and node_modules
            dirs[:] = [d for d in dirs if not d.startswith('.') and not d.startswith('__') and d != 'node_modules']

            
            for file in sorted(files):
                file_ext = os.path.splitext(file)[1].lower()
                if file_ext in SUPPORTED_EXTENSIONS:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, root_dir)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        language = SUPPORTED_EXTENSIONS[file_ext]
                        md_file.write(f"## File: `{relative_path}`\n\n")
                        md_file.write(f"```{language}\n{content}\n```\n\n")
                        md_file.write("---\n\n")
                        processed_files += 1
                    except UnicodeDecodeError:
                        print(f"Skipping binary file: {file_path}")
                    except Exception as e:
                        print(f"Error processing {file_path}: {str(e)}")
        
        md_file.write(f"\n\n> Total files processed: {processed_files}\n")
        print(f"Successfully generated code report at: {output_file}")

def generate_directory_tree(root_dir):
    """
    Generates a text representation of the directory tree
    """
    tree = []
    for root, dirs, files in os.walk(root_dir):
        # Skip node_modules
        dirs[:] = [d for d in dirs if d != 'node_modules']
        level = root.replace(root_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        tree.append(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            tree.append(f"{sub_indent}{f}")

    return '\n'.join(tree)

# Example usage:
if __name__ == "__main__":
    project_root = r"D:\Ayush\Internship In Action\Internship In Action\Backend"
    generate_code_report(project_root)