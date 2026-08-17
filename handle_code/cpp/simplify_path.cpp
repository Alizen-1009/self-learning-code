#include <bits/stdc++.h>
using namespace std;

// 假设输入是 Linux 的绝对路径，例如 /home/user/../tmp/./a。
string simplifyPath(const string& path) {
    vector<string> directories;
    int n = static_cast<int>(path.size());
    int i = 0;

    while (i < n) {
        // 跳过连续的 '/'
        while (i < n && path[i] == '/') {
            ++i;
        }

        int start = i;
        while (i < n && path[i] != '/') {
            ++i;
        }

        if (start == i) {
            continue;
        }

        string name = path.substr(start, i - start);
        if (name == ".") {
            continue;
        }
        if (name == "..") {
            if (!directories.empty()) {
                directories.pop_back();
            }
            continue;
        }

        directories.push_back(name);
    }

    if (directories.empty()) {
        return "/";
    }

    string result;
    for (const string& directory : directories) {
        result += "/" + directory;
    }
    return result;
}

int main() {
    vector<string> paths = {
        "/home/",
        "/home//foo/",
        "/home/user/../tmp/./a/",
        "/../",
        "/a/../../b/../c//.//"
    };

    for (const string& path : paths) {
        cout << path << " -> " << simplifyPath(path) << '\n';
    }
    return 0;
}
