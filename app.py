from flask import Flask, render_template, Response, redirect, url_for, request, session, flash, jsonify, send_file
from flask_session import Session
import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime, timedelta
import pickle
from pymongo import MongoClient
import certifi
from twilio.rest import Client
from functools import wraps
import csv
from io import StringIO, BytesIO
import urllib.parse
import atexit
import time

app = Flask(__name__)
app.secret_key = "your_secret_key_here"  # Replace with a secure key
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# MongoDB setup
MONGO_CONNECTION_STRING = "mongodb+srv://km0848230:5C4s8HSfEjfphlX5@cluster0.6rbbd.mongodb.net/mydatabase?retryWrites=true&w=majority"
client = MongoClient(MONGO_CONNECTION_STRING, tlsCAFile=certifi.where())
db = client["attendance"]

# Twilio setup
TWILIO_ACCOUNT_SID = "your_twilio_sid"  # Replace with your Twilio SID
TWILIO_AUTH_TOKEN = "your_twilio_auth_token"  # Replace with your Twilio Auth Token
TWILIO_PHONE_NUMBER = "whatsapp:+14155238886"  # Twilio WhatsApp number
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Global variables
capturing = False
fee = 100
required_days = 6
face_encodings_list = []
face_names = []
all_students = []
pickle_file = "face_encodings.pkl"
photo_dir = "static/"

# Ensure photo directory exists
if not os.path.exists(photo_dir):
    os.makedirs(photo_dir)

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            flash("يرجى تسجيل الدخول للوصول إلى هذه الصفحة", "warning")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Load face encodings
def load_face_encodings():
    global face_encodings_list, face_names
    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)
            face_encodings_list = data["encodings"]
            face_names = data["names"]
    else:
        for student in db.students.find():
            image_path = student.get("image_path")
            if image_path and os.path.exists(image_path):
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    face_encodings_list.append(encodings[0])
                    face_names.append(student["name"])
        with open(pickle_file, "wb") as f:
            pickle.dump({"encodings": face_encodings_list, "names": face_names}, f)
    if not face_encodings_list:
        print("⚠️ تحذير: لم يتم تحميل أي وجوه.")
    global all_students
    all_students = [student["name"] for student in db.students.find()]

# Send WhatsApp message
def send_whatsapp_message(to_number, message):
    try:
        twilio_client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=f"whatsapp:+{to_number}"
        )
        print(f"تم إرسال رسالة WhatsApp إلى {to_number}")
    except Exception as e:
        print(f"⚠️ خطأ في إرسال رسالة WhatsApp: {e}")

# Check absences
def check_absences():
    today = datetime.now().strftime("%Y-%m-%d")
    two_days_ago = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
    for student in all_students:
        last_attendance = db.attendance_records.find_one({"student": student}, sort=[("date", -1)])
        if not last_attendance or last_attendance["date"] < two_days_ago:
            student_data = db.students.find_one({"name": student})
            if student_data and student_data.get("phone_number"):
                send_whatsapp_message(
                    student_data["phone_number"],
                    f"تنبيه: الطالب {student} لم يحضر الجلسة لمدة يومين متتاليين."
                )

# Generate a fallback image
def generate_fallback_image():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "الكاميرا غير متاحة", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()

# Video feed generator
def gen_frames():
    global capturing
    max_retries = 3
    retry_delay = 2  # seconds
    for attempt in range(max_retries):
        video_cap = cv2.VideoCapture(0)  # Use confirmed working index 0
        if video_cap.isOpened():
            print("تم فتح الكاميرا عند الفهرس 0")
            break
        print(f"⚠️ فشل فتح الكاميرا عند الفهرس 0 (محاولة {attempt + 1}/{max_retries})")
        video_cap.release()
        if attempt < max_retries - 1:
            time.sleep(retry_delay)
    else:
        print("⚠️ خطأ: فشل فتح الكاميرا بعد جميع المحاولات")
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + generate_fallback_image() + b'\r\n')
        return
    video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    try:
        while True:
            ret, frame = video_cap.read()
            if not ret:
                print("⚠️ فشل في قراءة إطار من الكاميرا")
                video_cap.release()
                for attempt in range(max_retries):
                    video_cap = cv2.VideoCapture(0)
                    if video_cap.isOpened():
                        print(f"تم إعادة فتح الكاميرا عند الفهرس 0 (محاولة {attempt + 1})")
                        break
                    print(f"⚠️ فشل إعادة فتح الكاميرا (محاولة {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                if not video_cap.isOpened():
                    print("⚠️ خطأ: فشل إعادة فتح الكاميرا بعد جميع المحاولات")
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + generate_fallback_image() + b'\r\n')
                    break
                continue
            if capturing:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb)
                face_encodings = face_recognition.face_encodings(rgb, face_locations)
                face_names_detected = []
                today = datetime.now().strftime("%Y-%m-%d")
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(face_encodings_list, face_encoding, tolerance=0.5)
                    face_distances = face_recognition.face_distance(face_encodings_list, face_encoding)
                    if any(matches):
                        best_match = np.argmin(face_distances)
                        name = face_names[best_match]
                    else:
                        name = "Unknown"
                    face_names_detected.append(name)
                    if name != "Unknown" and name in all_students:
                        if not db.attendance_records.find_one({"student": name, "date": today}):
                            section = db.sections.find_one({"date": today}) or {"name": "غير محدد"}
                            db.attendance_records.insert_one({
                                "student": name,
                                "date": today,
                                "timestamp": datetime.now(),
                                "section_name": section["name"]
                            })
                            db.students.update_one({"name": name}, {"$inc": {"monthly_attendance": 1}})
                for (top, right, bottom, left), name in zip(face_locations, face_names_detected):
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Attendance Not Started", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("⚠️ فشل في تشفير الإطار")
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
    except Exception as e:
        print(f"⚠️ خطأ في gen_frames: {str(e)}")
    finally:
        video_cap.release()
        print("⚠️ تم تحرير الكاميرا في gen_frames")

# Camera status endpoint
@app.route('/camera_status')
@login_required
def camera_status():
    video_cap = cv2.VideoCapture(0)
    status = {"available": video_cap.isOpened()}
    if video_cap.isOpened():
        video_cap.release()
    return jsonify(status)

# Routes
@app.route('/')
@login_required
def home():
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return render_template('index.html', capturing=capturing, fee=fee, current_time=current_time)

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start')
@login_required
def start():
    global capturing
    capturing = True
    return redirect(url_for('home'))

@app.route('/end')
@login_required
def end():
    global capturing
    capturing = False
    check_absences()
    return redirect(url_for('home'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = db.users.find_one({"username": username})
        if user and user['password'] == password and user.get("active", True):
            session['user'] = username
            flash("تم تسجيل الدخول بنجاح", "success")
            return redirect(url_for('home'))
        else:
            flash("اسم المستخدم أو كلمة المرور غير صحيحة أو الحساب غير نشط", "danger")
    return render_template('login.html', current_year=datetime.now().year)

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("تم تسجيل الخروج", "info")
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
@login_required
def register():
    if session.get('user') != 'admin':
        flash("ليس لديك صلاحية للوصول إلى صفحة التسجيل", "danger")
        return redirect(url_for('home'))
    groups = list(db.groups.find())
    sections = list(db.sections.find())
    if request.method == 'POST':
        role = request.form.get('role')
        if role == 'student':
            student_name = request.form.get('student_name')
            phone_number = request.form.get('phone_number')
            level = request.form.get('level')
            group = request.form.get('group')
            image_file = request.files.get('student_image')
            if image_file and student_name:
                image_path = os.path.join(photo_dir, f"{student_name}.jpg")
                image_file.save(image_path)
                image = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    db.students.insert_one({
                        "name": student_name,
                        "phone_number": phone_number,
                        "level": level,
                        "group": group,
                        "monthly_attendance": 0,
                        "payment_status": False,
                        "payment_method": None,
                        "last_payment_date": None,
                        "image_path": image_path
                    })
                    global face_encodings_list, face_names, all_students
                    face_encodings_list.append(encodings[0])
                    face_names.append(student_name)
                    all_students.append(student_name)
                    with open(pickle_file, "wb") as f:
                        pickle.dump({"encodings": face_encodings_list, "names": face_names}, f)
                    flash("تم إضافة الطالب بنجاح ويمكن استخدام صورته لتسجيل الحضور", "success")
                    return redirect(url_for('home'))
                else:
                    flash("لم يتم العثور على وجه في الصورة", "danger")
            else:
                flash("يرجى إدخال جميع البيانات المطلوبة", "warning")
        else:
            username = request.form.get('username')
            password = request.form.get('password')
            if db.users.find_one({"username": username}):
                flash("اسم المستخدم موجود بالفعل", "warning")
            else:
                db.users.insert_one({
                    "username": username,
                    "password": password,
                    "role": role,
                    "register_date": datetime.now().strftime("%Y-%m-%d"),
                    "active": True
                })
                flash("تم إضافة المستخدم بنجاح", "success")
                return redirect(url_for('home'))
    return render_template('register.html', groups=groups, sections=sections)

@app.route('/manage_users')
@login_required
def manage_users():
    if session.get('user') != 'admin':
        flash("ليس لديك صلاحية الوصول إلى هذه الصفحة", "danger")
        return redirect(url_for('home'))
    users = {
        user["username"]: {
            "role": user["role"],
            "register_date": user.get("register_date", "غير معروف"),
            "active": user.get("active", True)
        } for user in db.users.find()
    }
    return render_template('manage_users.html', users=users)

@app.route('/edit_user/<username>', methods=['GET', 'POST'])
@login_required
def edit_user(username):
    if session.get('user') != 'admin':
        flash("ليس لديك صلاحية تنفيذ هذه العملية", "danger")
        return redirect(url_for('manage_users'))
    user = db.users.find_one({"username": username})
    if not user:
        flash("المستخدم غير موجود", "danger")
        return redirect(url_for('manage_users'))
    
    if request.method == 'POST':
        new_password = request.form.get('password')
        new_role = request.form.get('role')
        if new_password and new_role:
            db.users.update_one(
                {"username": username},
                {"$set": {
                    "password": new_password,
                    "role": new_role
                }}
            )
            flash(f"تم تعديل بيانات المستخدم {username} بنجاح", "success")
            return redirect(url_for('manage_users'))
        else:
            flash("يرجى ملء جميع الحقول", "warning")
    
    return render_template('edit_user.html', user=user)

@app.route('/toggle_user/<username>', methods=['POST'])
@login_required
def toggle_user(username):
    if session.get('user') != 'admin':
        flash("ليس لديك صلاحية تنفيذ هذه العملية", "danger")
        return redirect(url_for('manage_users'))
    user = db.users.find_one({"username": username})
    if not user:
        flash("المستخدم غير موجود", "danger")
        return redirect(url_for('manage_users'))
    if username == 'admin':
        flash("لا يمكن تعطيل المستخدم الإداري", "danger")
        return redirect(url_for('manage_users'))
    
    new_status = not user.get("active", True)
    db.users.update_one(
        {"username": username},
        {"$set": {"active": new_status}}
    )
    flash(f"تم {'تفعيل' if new_status else 'تعطيل'} المستخدم {username} بنجاح", "success")
    return redirect(url_for('manage_users'))

@app.route('/delete_user/<username>', methods=['POST'])
@login_required
def delete_user(username):
    if session.get('user') != 'admin':
        flash("ليس لديك صلاحية تنفيذ هذه العملية", "danger")
        return redirect(url_for('home'))
    if username == 'admin':
        flash("لا يمكن حذف المستخدم الإداري", "danger")
    elif db.users.find_one({"username": username}):
        db.users.delete_one({"username": username})
        flash("تم حذف المستخدم بنجاح", "success")
    else:
        flash("المستخدم غير موجود", "danger")
    return redirect(url_for('manage_users'))

@app.route('/mark_paid/<student>')
@login_required
def mark_paid(student):
    student_data = db.students.find_one({"name": student})
    if not student_data:
        flash("الطالب غير موجود", "error")
        return redirect(url_for('attendance_status'))
    
    db.students.update_one({"name": student}, {"$set": {
        "payment_status": True,
        "payment_method": "Manual",
        "last_payment_date": datetime.now().strftime("%Y-%m-%d")
    }})
    flash(f"تم تسجيل دفع الطالب {student} بنجاح", "success")
    return redirect(url_for('attendance_status'))

@app.route('/mark_unpaid/<student>')
@login_required
def mark_unpaid(student):
    student_data = db.students.find_one({"name": student})
    if not student_data:
        flash("الطالب غير موجود", "error")
        return redirect(url_for('attendance_status'))
    
    db.students.update_one({"name": student}, {"$set": {
        "payment_status": False,
        "payment_method": None,
        "last_payment_date": None
    }})
    flash(f"تم إلغاء دفع الطالب {student}", "warning")
    return redirect(url_for('attendance_status'))

@app.route('/send_manual_reminder/<student>')
@login_required
def send_manual_reminder(student):
    student_data = db.students.find_one({"name": student})
    if not student_data:
        flash("الطالب غير موجود", "error")
        return redirect(url_for('attendance_status'))
    
    if student_data.get("phone_number"):
        send_whatsapp_message(
            student_data["phone_number"],
            f"تنبيه يدوي: الطالب {student}، يرجى مراجعة حالة الحضور أو الدفع."
        )
        flash(f"تم إرسال تنبيه يدوي للطالب {student}", "success")
    else:
        flash("رقم الهاتف غير متوفر لهذا الطالب", "error")
    return redirect(url_for('attendance_status'))

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        username = request.form.get('username')
        user = db.users.find_one({"username": username})
        if user:
            flash("تم إرسال طلب إعادة تعيين كلمة المرور. يرجى التواصل مع الإدارة.", "success")
            student_data = db.students.find_one({"name": username})
            if student_data and student_data.get("phone_number"):
                send_whatsapp_message(
                    student_data["phone_number"],
                    f"تم طلب إعادة تعيين كلمة المرور للمستخدم {username}. يرجى التواصل مع الإدارة."
                )
        else:
            flash("اسم المستخدم غير موجود", "danger")
        return redirect(url_for('login'))
    return render_template('forgot_password.html')

@app.route('/student_profile/<student_name>')
@login_required
def student_profile(student_name):
    student_data = db.students.find_one({"name": student_name})
    if not student_data:
        flash("الطالب غير موجود", "danger")
        return redirect(url_for('attendance_status'))
    
    attendance_records = list(db.attendance_records.find({"student": student_name}))
    return render_template('student_profile.html', student=student_data, attendance=attendance_records)

@app.route('/attendance_status', methods=['GET', 'POST'])
@login_required
def attendance_status():
    groups = list(db.groups.find())
    sections = list(db.sections.find())
    selected_group = request.args.get('group', 'all')
    search_query = request.args.get('search', '').strip()
    selected_month = request.args.get('month', 'all')
    current_date = datetime.now().strftime("%Y-%m-%d")

    months = [
        {"name": "يناير", "value": "01"}, {"name": "فبراير", "value": "02"},
        {"name": "مارس", "value": "03"}, {"name": "أبريل", "value": "04"},
        {"name": "مايو", "value": "05"}, {"name": "يونيو", "value": "06"},
        {"name": "يوليو", "value": "07"}, {"name": "أغسطس", "value": "08"},
        {"name": "سبتمبر", "value": "09"}, {"name": "أكتوبر", "value": "10"},
        {"name": "نوفمبر", "value": "11"}, {"name": "ديسمبر", "value": "12"}
    ]

    monthly_data = []
    total_students = len(all_students)
    total_attendance_days = len(db.attendance_records.distinct("date"))
    paid_count = db.students.count_documents({"payment_status": True})
    unpaid_count = total_students - paid_count
    total_fees = total_students * fee
    total_paid = paid_count * fee
    remaining_amount = total_fees - total_paid

    attendance_sum = 0
    filtered_students = []

    for student in all_students:
        student_data = db.students.find_one({"name": student})
        if student_data:
            group_match = selected_group == 'all' or student_data.get("group") == selected_group
            search_match = not search_query or search_query.lower() in student.lower()
            if group_match and search_match:
                filtered_students.append(student)
                attendance_sum += student_data.get("monthly_attendance", 0)

    average_attendance = attendance_sum / len(filtered_students) if filtered_students else 0
    payment_percentage = (paid_count / total_students * 100) if total_students > 0 else 0

    for student in filtered_students:
        student_data = db.students.find_one({"name": student})
        if student_data:
            count = student_data.get("monthly_attendance", 0)
            last_payment_date = student_data.get("last_payment_date", None)
            if selected_month != 'all' and last_payment_date:
                if not last_payment_date.startswith(f"{datetime.now().year}-{selected_month}"):
                    continue
            pay_status = "Paid" if student_data.get("payment_status", False) else "Not Paid"
            monthly_data.append({
                "name": student,
                "count": count,
                "message": f"لم يكمل الشهر ({count}/{required_days})" if count < required_days else "",
                "fee": fee,
                "paid": pay_status,
                "method": student_data.get("payment_method", "--"),
                "last_payment_date": last_payment_date or "--",
                "group": student_data.get("group", "غير محدد"),
                "phone_number": student_data.get("phone_number", "--")
            })

    return render_template('attendance.html', monthly_data=monthly_data, total_students=total_students,
                           total_attendance_days=total_attendance_days, paid_count=paid_count,
                           unpaid_count=unpaid_count, total_fees=total_fees, total_paid=total_paid,
                           remaining_amount=remaining_amount, average_attendance=round(average_attendance, 2),
                           payment_percentage=round(payment_percentage, 2), required_days=required_days,
                           groups=groups, sections=sections, selected_group=selected_group,
                           search_query=search_query, current_date=current_date,
                           selected_month=selected_month, months=months)

@app.route('/add_section', methods=['GET', 'POST'])
@login_required
def add_section():
    if session.get('user') != 'admin':
        flash("ليس لديك صلاحية للوصول إلى هذه الصفحة", "danger")
        return redirect(url_for('home'))
    if request.method == 'POST':
        add_type = request.form.get('add_type')
        print(f"Add type: {add_type}")
        if add_type == 'session':
            section_name = request.form.get('section_name')
            section_date = request.form.get('section_date')
            print(f"Session: {section_name}, {section_date}")
            if section_name and section_date:
                db.sections.insert_one({"name": section_name, "date": section_date})
                flash("تم إضافة الجلسة بنجاح", "success")
            else:
                flash("يرجى ملء جميع حقول الجلسة", "danger")
        elif add_type == 'group':
            group_name = request.form.get('group_name')
            group_days = request.form.get('group_days').split(',') if request.form.get('group_days') else []
            group_time = request.form.get('group_time')
            print(f"Group: {group_name}, {group_days}, {group_time}")
            if group_name and group_time:
                db.groups.insert_one({"name": group_name, "days": group_days, "time": group_time})
                flash("تم إضافة المجموعة بنجاح", "success")
            else:
                flash("يرجى ملء جميع حقول المجموعة", "danger")
        return redirect(url_for('home'))
    return render_template('add_section.html')

@app.route('/total_payment')
@login_required
def total_payment():
    paid_count = db.students.count_documents({"payment_status": True})
    total = paid_count * fee
    return render_template('total_payment.html', paid_count=paid_count, fee=fee, total=total)

@app.route('/set_fee', methods=['GET', 'POST'])
@login_required
def set_fee():
    global fee
    if request.method == 'POST':
        try:
            fee = float(request.form.get('fee'))
            flash("تم تعديل الرسوم بنجاح", "success")
        except (ValueError, TypeError):
            fee = 100
            flash("خطأ في إدخال الرسوم، تم إعادة تعيينها إلى 100", "warning")
        return redirect(url_for('home'))
    return render_template('set_fee.html', fee=fee)

@app.route('/send_message', methods=['GET', 'POST'])
@login_required
def send_message():
    if session.get('user') != 'admin':
        flash("ليس لديك صلاحية للوصول إلى هذه الصفحة", "danger")
        return redirect(url_for('home'))
    if request.method == 'POST':
        message = request.form.get('message')
        today = datetime.now().strftime("%Y-%m-%d")
        for student in all_students:
            if not db.attendance_records.find_one({"student": student, "date": today}):
                student_data = db.students.find_one({"name": student})
                if student_data and student_data.get("phone_number"):
                    send_whatsapp_message(student_data["phone_number"], message)
        flash("تم إرسال الرسالة لجميع الطلاب المتغيبين", "success")
        return redirect(url_for('home'))
    return render_template('send_message.html')

@app.route('/send_reminders')
@login_required
def send_reminders():
    for student in all_students:
        student_data = db.students.find_one({"name": student})
        if student_data and not student_data.get("payment_status", False) and student_data.get("phone_number"):
            send_whatsapp_message(
                student_data["phone_number"],
                f"تنبيه: الطالب {student} لم يسدد الرسوم المستحقة ({fee} ر.س). يرجى التسديد في أقرب وقت."
            )
    flash("تم إرسال تنبيهات الدفع للطلاب غير المدفوعين", "success")
    return redirect(url_for('attendance_status'))

@app.route('/export_attendance')
@login_required
def export_attendance():
    monthly_data = []
    for student in all_students:
        student_data = db.students.find_one({"name": student})
        if student_data:
            count = student_data.get("monthly_attendance", 0)
            pay_status = "Paid" if student_data.get("payment_status", False) else "Not Paid"
            monthly_data.append({
                "name": student,
                "count": count,
                "fee": fee,
                "paid": pay_status,
                "method": student_data.get("payment_method", "--"),
                "last_payment_date": student_data.get("last_payment_date", "--"),
                "group": student_data.get("group", "غير محدد"),
                "phone_number": student_data.get("phone_number", "--")
            })

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["اسم الطالب", "المجموعة/الجلسة", "عدد أيام الحضور", "مبلغ الرسوم", "حالة الدفع", "طريقة الدفع", "تاريخ آخر دفع", "رقم الهاتف"])
    for data in monthly_data:
        writer.writerow([data["name"], data["group"], data["count"], data["fee"], data["paid"], data["method"], data["last_payment_date"], data["phone_number"]])
    
    output.seek(0)
    return send_file(
        BytesIO(output.getvalue().encode('utf-8')),
        mimetype="text/csv",
        as_attachment=True,
        download_name="attendance_report.csv"
    )

@app.route('/reset_attendance')
@login_required
def reset_attendance():
    if session.get('user') != 'admin':
        flash("ليس لديك صلاحية لإعادة تعيين الحضور", "danger")
        return redirect(url_for('attendance_status'))
    db.students.update_many({}, {"$set": {"monthly_attendance": 0}})
    db.attendance_records.delete_many({})
    flash("تم إعادة تعيين بيانات الحضور بنجاح", "success")
    return redirect(url_for('attendance_status'))

@app.route('/shutdown')
def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    if func:
        func()
    return "Shutting down..."

# Cleanup on exit
def cleanup():
    print("⚠️ تحرير الكاميرا...")

atexit.register(cleanup)

# Load face encodings on startup
load_face_encodings()

if __name__ == '__main__':
    if not db.users.find_one({"username": "admin"}):
        db.users.insert_one({
            "username": "admin",
            "password": "admin123",
            "role": "admin",
            "register_date": datetime.now().strftime("%Y-%m-%d"),
            "active": True
        })
    if not db.users.find_one({"username": "user1"}):
        db.users.insert_one({
            "username": "user1",
            "password": "password1",
            "role": "user",
            "register_date": datetime.now().strftime("%Y-%m-%d"),
            "active": True
        })
    app.run(host='0.0.0.0', port=5000, debug=False)