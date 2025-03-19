from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from benchmark.models import BenchmarkTask

User = get_user_model()

class Command(BaseCommand):
    help = 'Creates initial benchmark tasks'
    
    def handle(self, *args, **options):
        # Get or create admin user
        admin_user = User.objects.filter(is_superuser=True).first()
        if not admin_user:
            self.stdout.write(self.style.WARNING('No admin user found. Tasks will have no creator.'))
        
        # Create reasoning task
        reasoning_task, created = BenchmarkTask.objects.get_or_create(
            name='Reasoning and Problem Solving',
            defaults={
                'description': 'Tests the model\'s ability to solve problems and reason through complex scenarios.',
                'category': 'reasoning',
                'prompts': [
                    "Explain why ice floats on water using physics principles.",
                    "A bat and ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
                    "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
                    "You have a 3-gallon jug and a 5-gallon jug. How can you measure exactly 4 gallons of water?",
                    "Explain the concept of diminishing returns using a real-world example."
                ],
                'reference_answers': [
                    "Ice floats on water because it is less dense than liquid water. When water freezes, the molecules arrange in a crystalline structure that takes up more space than liquid water, making it less dense.",
                    "The ball costs $0.05. If the ball costs x, then the bat costs x + $1.00. Together they cost $1.10, so x + (x + $1.00) = $1.10. This simplifies to 2x + $1.00 = $1.10, so 2x = $0.10, and x = $0.05.",
                    "5 minutes. The rate of production scales linearly with the number of machines, so 100 machines would make 100 widgets in the same time it takes 5 machines to make 5 widgets.",
                    "Fill the 5-gallon jug completely. Pour from the 5-gallon jug into the 3-gallon jug until the 3-gallon jug is full. This leaves 2 gallons in the 5-gallon jug. Empty the 3-gallon jug. Pour the 2 gallons from the 5-gallon jug into the 3-gallon jug. Fill the 5-gallon jug again. Pour from the 5-gallon jug into the 3-gallon jug until it's full, which requires 1 gallon (since the 3-gallon jug already has 2 gallons). This leaves 4 gallons in the 5-gallon jug.",
                    "Diminishing returns occurs when adding more of a factor of production results in smaller increases in output. For example, in a restaurant, adding more cooks initially increases food production significantly. However, after a certain point, adding more cooks provides less benefit because they start getting in each other's way or have to share limited kitchen space and equipment."
                ],
                'metrics': ['response_time_ms', 'token_efficiency', 'lexical_f1'],
                'created_by': admin_user,
                'is_public': True
            }
        )
        
        if created:
            self.stdout.write(self.style.SUCCESS(f'Created task: {reasoning_task.name}'))
        else:
            self.stdout.write(f'Task already exists: {reasoning_task.name}')
        
        # Create factual knowledge task
        factual_task, created = BenchmarkTask.objects.get_or_create(
            name='Factual Knowledge',
            defaults={
                'description': 'Tests the model\'s ability to recall factual information accurately.',
                'category': 'factual',
                'prompts': [
                    "List the capitals of the G7 countries.",
                    "What are the first 10 elements in the periodic table?",
                    "Who wrote 'Pride and Prejudice' and in what year was it published?",
                    "What is the tallest mountain in the world and how tall is it?",
                    "Name the planets in our solar system in order from the sun."
                ],
                'reference_answers': [
                    "The capitals of the G7 countries are: Washington D.C. (USA), Ottawa (Canada), London (UK), Paris (France), Berlin (Germany), Rome (Italy), and Tokyo (Japan).",
                    "The first 10 elements in the periodic table are: Hydrogen (H), Helium (He), Lithium (Li), Beryllium (Be), Boron (B), Carbon (C), Nitrogen (N), Oxygen (O), Fluorine (F), and Neon (Ne).",
                    "Jane Austen wrote 'Pride and Prejudice', which was published in 1813.",
                    "Mount Everest is the tallest mountain in the world, standing at 8,848.86 meters (29,031.7 feet) above sea level.",
                    "The planets in our solar system in order from the sun are: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune."
                ],
                'metrics': ['response_time_ms', 'token_efficiency', 'lexical_f1'],
                'created_by': admin_user,
                'is_public': True
            }
        )
        
        if created:
            self.stdout.write(self.style.SUCCESS(f'Created task: {factual_task.name}'))
        else:
            self.stdout.write(f'Task already exists: {factual_task.name}')
        
        # Create instruction following task
        instruction_task, created = BenchmarkTask.objects.get_or_create(
            name='Instruction Following',
            defaults={
                'description': 'Tests the model\'s ability to follow specific instructions accurately.',
                'category': 'instruction',
                'prompts': [
                    "Summarize this paragraph in exactly 3 bullet points: The Industrial Revolution was a period of major industrialization and innovation that took place during the late 1700s and early 1800s. The Industrial Revolution began in Great Britain and quickly spread throughout the world. The American Industrial Revolution commonly referred to as the Second Industrial Revolution, started in the late 1800s. The invention of steam-powered engines and the cotton gin were significant inventions of this era that changed manufacturing forever.",
                    "Write a haiku about artificial intelligence.",
                    "Translate this sentence into French, Spanish, and German: 'The quick brown fox jumps over the lazy dog.'",
                    "Explain quantum computing to a 10-year-old child.",
                    "Create a 5-item to-do list for someone planning a birthday party."
                ],
                'metrics': ['response_time_ms', 'token_efficiency'],
                'created_by': admin_user,
                'is_public': True
            }
        )
        
        if created:
            self.stdout.write(self.style.SUCCESS(f'Created task: {instruction_task.name}'))
        else:
            self.stdout.write(f'Task already exists: {instruction_task.name}')
        
        # Create code task
        code_task, created = BenchmarkTask.objects.get_or_create(
            name='Code Understanding and Generation',
            defaults={
                'description': 'Tests the model\'s ability to understand and generate code.',
                'category': 'code',
                'prompts': [
                    "Write a Python function to check if a string is a palindrome.",
                    "Explain what this code does: `const memoize = fn => { const cache = {}; return (...args) => { const key = JSON.stringify(args); return cache[key] = cache[key] || fn(...args); }; };`",
                    "Write a SQL query to find the top 5 customers who have spent the most money, given tables 'customers' (id, name, email) and 'orders' (id, customer_id, amount, date).",
                    "Debug this Python code: ```def fibonacci(n): if n <= 0: return 0 elif n == 1: return 1 else: return fibonacci(n+1) - fibonacci(n+2)```",
                    "Write a simple HTML form with fields for name, email, and a submit button. Include CSS to style the form with a width of 300px and centered on the page."
                ],
                'metrics': ['response_time_ms', 'token_efficiency'],
                'created_by': admin_user,
                'is_public': True
            }
        )
        
        if created:
            self.stdout.write(self.style.SUCCESS(f'Created task: {code_task.name}'))
        else:
            self.stdout.write(f'Task already exists: {code_task.name}')
        
        self.stdout.write(self.style.SUCCESS('Benchmark tasks created successfully!')) 