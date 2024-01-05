import requests
import random
import os

def get_org_users(org_name, token, role):
    users = []
    page = 1
    while True:
        url = f"https://api.github.com/orgs/{org_name}/members?role={role}&per_page=100&page={page}"
        headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            batch = response.json()
            if not batch:
                break
            users.extend([user['login'] for user in batch])
            page += 1
        else:
            print(f"Failed to fetch organization {role}s: {response.status_code}, {response.text}")
            break
    return users

def assign_issue(issue_number, assignee, repo_name, token):
    url = f"https://api.github.com/repos/{repo_name}/issues/{issue_number}/assignees"
    headers = {
        'Authorization': f'token {token}',
        'Accept': 'application/vnd.github.v3+json'
    }
    data = {'assignees': [assignee]}
    response = requests.post(url, json=data, headers=headers)
    if response.status_code not in [200, 201]:
        print(f"Failed to assign issue: {response.status_code}, {response.text}")

def main():
    token = os.getenv('GITHUB_TOKEN')
    org_name = os.getenv('ORG_NAME')
    repo_name = os.getenv('REPO_NAME')
    issue_number = os.getenv('ISSUE_NUMBER')

    members = get_org_users(org_name, token, 'member')
    owners = get_org_users(org_name, token, 'admin')
    all_users = list(set(members + owners))

    if not all_users:
        print("No users found in the organization.")
        return

    assignee = random.choice(all_users)
    assign_issue(issue_number, assignee, repo_name, token)
    print(f"Issue {issue_number} has been assigned to {assignee}")

if __name__ == "__main__":
    main()
