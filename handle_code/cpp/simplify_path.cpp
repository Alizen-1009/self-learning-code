/*
中文说明：把 Linux 绝对路径规范化：合并连续斜杠，忽略 '.'，用 '..' 返回上级目录，
且不会越过根目录。
解题方法：逐段扫描路径，用 vector 充当目录栈；普通目录入栈，'..' 弹栈，最后用
'/' 重新拼接。复杂度 O(|path|) 时间，O(|path|) 空间。main 中给出若干演示样例。

English: Canonicalize a Linux absolute path by collapsing slashes, ignoring '.',
and applying '..' without moving above root. Scan components and use a vector as
a directory stack, then join it with '/'. O(|path|) time and space; main contains
demonstration cases.
*/
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
