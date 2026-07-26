#include<bits/stdc++.h>
#define ll long long
using namespace std;

const int N = 5050;
int p[N], d[N], q[N], cnt[N];
bool alive[N];
int n;

void solve() {
    cin >> n;
    for (int i = 1; i <= n; i++) cin >> p[i];
    for (int i = 1; i <= n; i++) cin >> d[i];

    for (int i = 1; i <= n; i++) {
        cnt[i] = 0;
        alive[i] = true;
        for (int j = i + 1; j <= n; j++) {
            if (p[j] > p[i]) cnt[i]++;
        }
    }

    for (int val = 1; val <= n; val++) {
        int best = 0;
        for (int i = 1; i <= n; i++) {
            if (!alive[i] || cnt[i] != d[i]) continue;
            if (best == 0 || p[i] < p[best]) best = i;
        }

        if (best == 0) {
            cout << -1 << '\n';
            return;
        }

        q[best] = val;
        alive[best] = false;
        for (int i = 1; i < best; i++) {
            if (alive[i] && p[i] < p[best]) cnt[i]--;
        }
    }

    for (int i = 1; i <= n; i++) {
        cout << q[i] << " \n"[i == n];
    }
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int T = 1;
    cin >> T;
    while (T--) solve();

    return 0;
}