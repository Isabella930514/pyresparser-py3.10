import os
from packaging import version

# List of paths to the requirements.txt files you want to merge
file_paths = ['C:/Users/cjv2124/pyresparser-py3.10/requirements.txt',
              'C:/Users/cjv2124/pyresparser-py3.10/pyresparser/requirements.txt']

# Dictionary to store the highest version of each package
package_versions = {}


# Function to parse the package name and version from a line
def parse_package(line):
    parts = line.strip().split('==')
    if len(parts) == 2:
        pkg, ver = parts
        return pkg, version.parse(ver)
    return None, None


# Read each file
for file_path in file_paths:
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            for line in file:
                pkg, ver = parse_package(line)
                if pkg:
                    if pkg in package_versions:
                        # Only update if the version is greater
                        if ver > package_versions[pkg]:
                            package_versions[pkg] = ver
                    else:
                        package_versions[pkg] = ver

# Write the highest versions of each package to a new file
with open('optimized_requirements.txt', 'w') as file:
    for pkg, ver in sorted(package_versions.items()):
        file.write(f'{pkg}=={ver}\n')

print('Optimized requirements.txt has been created.')
